from typing import Optional

import logging
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional
import os
import re
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from fairseq import utils
from fairseq.models import register_model, register_model_architecture, FairseqEncoder, BaseFairseqModel
from fairseq.models.transformer import TransformerModel, Embedding
from fairseq.models.bart import BARTModel
from fairseq.checkpoint_utils import load_checkpoint_to_cpu, load_pretrained_component_from_model
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from .hub_interface import BARTHubInterface
import json
import pickle as pkl
from torch_geometric.nn.conv.rgcn_conv import RGCNConv


logger = logging.getLogger(__name__)


EDGE_TYPES = [58, 172]
def _edge_list(kg, n_entity, hop=2, cnt_threshold=1000):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != 185:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > cnt_threshold and r not in relation_idx:
            relation_idx[r] = len(relation_idx)
    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > cnt_threshold], len(relation_idx)


class GCNModelWithContextEmbed(BaseFairseqModel):
    def __init__(self, config, vocab_dict):
        super().__init__()
        self.kg_dim = config.kg_dim
        self.n_entity = config.n_entity
        self.n_relation = config.n_relation
        self.n_bases = config.n_bases
        self.n_movie = config.n_movie
        self.rgcn_n_hop = config.rgcn_n_hop
        self.r_thres = config.relation_count_threshold
        self.movie_start_idx = config.movie_start_idx

        if config.movieid2entityid_json is None:  # this means that movie id is the same as entity id
            movieid2entityid = None
        else:
            with open(config.movieid2entityid_json, 'r') as f:
                movieid2entityid = json.load(f)
        self.idx2entityid = torch.zeros(self.n_movie, dtype=torch.long).fill_(-1)
        if torch.cuda.is_available():
            self.idx2entityid = self.idx2entityid.cuda()
        for idx in range(self.n_movie):
            movie_idx = idx + self.movie_start_idx
            movie_id = vocab_dict[movie_idx]
            if movieid2entityid is None:
                entity_id = int(re.sub('\D', '', movie_id))  # remove the beginning "@" if exists
                self.idx2entityid[idx] = entity_id
            elif movie_id in movieid2entityid:
                entity_id = movieid2entityid[movie_id]
                self.idx2entityid[idx] = entity_id

        self.kg = pkl.load(open(config.kg_file, "rb"))
        edge_list, self.n_relation = _edge_list(self.kg, self.n_entity, hop=self.rgcn_n_hop, cnt_threshold=self.r_thres)
        edge_list = list(set(edge_list))

        self.dbpedia_edge_sets = torch.LongTensor(edge_list)
        if torch.cuda.is_available():
            self.dbpedia_edge_sets = self.dbpedia_edge_sets.cuda()
        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        self.db_edge_type = self.dbpedia_edge_sets[:, 2]
        self.dbpedia_RGCN = RGCNConv(self.n_entity, self.kg_dim, self.n_relation, self.n_bases)

        self.time_weight = config.time_weight
        self.time_attn = torch.pow(self.time_weight, torch.arange(100, dtype=torch.float))
        if torch.cuda.is_available():
            self.time_attn = self.time_attn.cuda()

        self.use_text_entities = getattr(config, 'use_text_entities', False)
        self.context_rep_weight = getattr(config, 'context_rep_weight', 0)
        self.recommend_only_kg_rep = getattr(config, 'recommend_only_kg_rep', False)
        self.recommend_only_context_rep = getattr(config, 'recommend_only_context_rep', False)

        self.entity_is_movie = torch.zeros(self.n_entity).to(self.idx2entityid)
        for entityid in self.idx2entityid:
            if entityid != -1:
                self.entity_is_movie[entityid] = 1

        self.context_to_entity = nn.Linear(config.encoder_embed_dim, self.n_entity)
        self.output_bias = nn.Parameter(torch.zeros(self.n_entity))
        nn.init.normal_(self.output_bias.data)

    def forward(self, net_input, context_reps=None):
        if self.recommend_only_context_rep:
            assert context_reps is not None
            bart_scores = self.context_to_entity(context_reps)  # B x De -> B x E
            return None, bart_scores

        movie_sets = net_input["movie_sets"]
        movie_idxs = net_input["movie_idxs"]
        text_entities = net_input["text_entities"]
        db_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type)  # E x D

        if context_reps is not None:  # use context representations
            bart_scores = self.context_to_entity(context_reps)  # B x De -> B x E
        else:
            bart_scores = None

        user_representations = []
        for i, mvs in enumerate(movie_idxs):
            if self.use_text_entities:
                entityids = text_entities[i]
            else:
                entityids = self.idx2entityid[mvs]

            if len(entityids) == 0:
                user_representation = torch.zeros(self.kg_dim).to(movie_sets)
            else:
                entityids = torch.masked_select(entityids, entityids != -1)
                if len(entityids) == 0:
                    user_representation = torch.zeros(self.kg_dim).to(movie_sets)
                else:
                    user_representation = db_nodes_features[entityids]
                    e_num = len(user_representation)
                    attn_weight = self.time_attn[:e_num]/self.time_attn[:e_num].sum()
                    if e_num > 100:
                        user_representation = user_representation[-100:]
                    user_representation = torch.matmul(attn_weight, user_representation)  # N, N x D -> D
            user_representations.append(user_representation)
        user_representations = torch.stack(user_representations)  # B x D
        movie_scores = F.linear(user_representations, db_nodes_features, self.output_bias)  # B x E

        return movie_scores, bart_scores

    def recommend(self, net_input, context_reps=None):
        movie_scores, bart_scores = self.forward(net_input, context_reps)
        if movie_scores is None:
            return F.softmax(bart_scores, dim=1)
        elif bart_scores is None:
            return F.softmax(movie_scores, dim=1)
        else:
            return (1 - self.context_rep_weight) * F.softmax(movie_scores, dim=1) + \
                   self.context_rep_weight * F.softmax(bart_scores, dim=1)


class EncoderWithRec(FairseqEncoder):
    def __init__(self, cfg, conv_encoder):
        self.cfg = cfg
        self.cfg.eos_as_context_rep = getattr(cfg, 'eos_as_context_rep', False)
        self.cfg.recommend_only_kg_rep = getattr(cfg, 'recommend_only_kg_rep', False)
        self.cfg.recommend_only_context_rep = getattr(cfg, 'recommend_only_context_rep', False)
        self.cfg.entity_output = True

        super().__init__(conv_encoder.dictionary)
        self.conv_encoder = conv_encoder

        self.recommender = GCNModelWithContextEmbed(cfg, conv_encoder.dictionary)
        self.idx2entityid = self.recommender.idx2entityid
        self.entity_is_movie = self.recommender.entity_is_movie

    def max_positions(self):
        return self.conv_encoder.max_source_positions

    def forward(self, net_input):
        # get conv encoder results
        encoder_out = self.conv_encoder(
            net_input['src_tokens'],
            src_lengths=net_input['src_lengths'])
        # get context representations
        if self.cfg.recommend_only_kg_rep:
            context_reps = None
        else:
            context_reps = encoder_out["encoder_out"][0].transpose(0, 1)  # B x T x C
            if self.cfg.eos_as_context_rep:
                context_reps = context_reps[:, -1, :].clone()  # B x C
            else:
                context_reps = context_reps.masked_fill(encoder_out["encoder_padding_mask"][0].unsqueeze(-1), 0)
                context_reps = context_reps.sum(dim=1) / encoder_out["src_lengths"][0]  # B x C
        # get recommendation results
        movie_scores = self.recommender(net_input, context_reps)
        encoder_out['movie_scores'] = movie_scores
        return encoder_out

    def recommend(self, net_input):
        if self.cfg.recommend_only_kg_rep:
            context_reps = None
        else:
            encoder_out = self.conv_encoder(
                net_input['src_tokens'],
                src_lengths=net_input['src_lengths'])
            context_reps = encoder_out["encoder_out"][0].transpose(0, 1)  # B x T x C
            if self.cfg.eos_as_context_rep:
                context_reps = context_reps[:, -1, :].clone()  # B x C
            else:
                context_reps = context_reps.masked_fill(encoder_out["encoder_padding_mask"][0].unsqueeze(-1), 0)
                context_reps = context_reps.sum(dim=1) / encoder_out["src_lengths"][0]  # B x C
        return self.recommender.recommend(net_input, context_reps)

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        new_movie_scores = encoder_out['movie_scores'].index_select(0, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
            "movie_scores": new_movie_scores,  # B x M
        }


@register_model("convrec_time_bart")
class ConvRecBARTModel(BARTModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        if getattr(args, 'with_recommender', False):
            self.encoder = EncoderWithRec(args, self.encoder)
            self.idx2entityid = self.encoder.idx2entityid
            self.entity_is_movie = self.encoder.entity_is_movie
            self.entity_output = True

    @staticmethod
    def add_args(parser):
        super(ConvRecBARTModel, ConvRecBARTModel).add_args(parser)
        parser.add_argument(
            "--bart-dict-file",
            type=str,
            help="dict file name for original bart",
        )
        parser.add_argument(
            "--bart-model-file",
            type=str,
            help="model path for original bart",
        )
        parser.add_argument(
            "--load-finetuned-bart-from",
            type=str,
            help="model path for finetuned bart",
        )
        parser.add_argument(
            "--load-rec-bart-from",
            type=str,
            help="model path for bart with recommendation",
        )
        parser.add_argument(
            "--with-recommender",
            action="store_true",
            help="add a recommender in encoder",
        )
        parser.add_argument(
            "--kg-dim",
            type=int,
            default=128,
            help="hidden dimension for gcn",
        )
        parser.add_argument(
            "--n-entity",
            type=int,
            default=64368,
            help="entity number",
        )
        parser.add_argument(
            "--n-relation",
            type=int,
            default=46,
            help="relation type number",
        )
        parser.add_argument(
            "--n-bases",
            type=int,
            default=8,
            help="the number of bases",
        )
        parser.add_argument(
            "--n-movie",
            type=int,
            default=6637,
            help="movie number",
        )
        parser.add_argument(
            "--relation-count-threshold",
            type=int,
            default=1000,
            help="only consider the relations that appear more than this value",
        )
        parser.add_argument(
            "--rgcn-n-hop",
            type=int,
            default=2,
            help="the number of hops in rgcn",
        )
        parser.add_argument(
            "--movie-start-idx",
            type=int,
            default=50263,
            help="movie start index",
        )
        parser.add_argument(
            "--movieid2entityid-json",
            type=str,
            default="movie_id2entityid.json",
            help="path for the json file with movieid2entityid dictionary",
        )
        parser.add_argument(
            "--use-text-entities",
            action="store_true",
            help="use text entities as part of RGCN input",
        )
        parser.add_argument(
            "--kg-file",
            type=str,
            default="kgdata/redial.pkl",
            help="path for kg file",
        )
        parser.add_argument(
            "--eos-as-context-rep",
            action="store_true",
            help="if True use final eos token's representation as context rep, if False use average representation",
        )
        parser.add_argument(
            "--recommend-only-context-rep",
            action="store_true",
            help="if True only use context representation for recommendation",
        )
        parser.add_argument(
            "--recommend-only-kg-rep",
            action="store_true",
            help="if True only use kg's representation for recommendation",
        )
        parser.add_argument(
            "--time-weight",
            type=float,
            default=1.5,
            help="the base for time-attention",
        )
        parser.add_argument(
            "--context-rep-weight",
            type=float,
            default=0,
            help="if not None, then 'cos' or value < 1, serving as the weight added to movie reps",
        )

    @classmethod
    def build_model(cls, cfg, task):
        model = super().build_model(cfg, task)
        if getattr(cfg, 'bart_model_file', '') != '':
            # load the original bart dictionary
            paths = utils.split_paths(cfg.data)
            original_dict = task.load_dictionary(os.path.join(paths[0], cfg.bart_dict_file))
            # load bart model except for the embeddings, since vocab sizes are not the same
            state = load_checkpoint_to_cpu(cfg.bart_model_file)
            encoder_state_dict, decoder_state_dict = OrderedDict(), OrderedDict()
            for key in state["model"].keys():
                if key.startswith('encoder') and 'embed_tokens' not in key:
                    # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
                    component_subkey = key[len('encoder')+1:]
                    encoder_state_dict[component_subkey] = state["model"][key]
                elif key.startswith('decoder') and 'embed_tokens' not in key and 'output_projection' not in key:
                    # decoder.input_layers.0.0.weight --> input_layers.0.0.weight
                    component_subkey = key[len('decoder')+1:]
                    decoder_state_dict[component_subkey] = state["model"][key]
            if isinstance(model.encoder, EncoderWithRec):
                model.encoder.conv_encoder.load_state_dict(encoder_state_dict, strict=False)
            else:
                model.encoder.load_state_dict(encoder_state_dict, strict=False)
            model.decoder.load_state_dict(decoder_state_dict, strict=False)
            # deal with embeddings
            embed_state = state['model']['encoder.embed_tokens.weight'][:len(original_dict), :]
            cur_weight = model.decoder.embed_tokens.weight.detach().numpy()
            cur_weight[:len(original_dict)] = embed_state
            if isinstance(model.encoder, EncoderWithRec):
                model.encoder.conv_encoder.embed_tokens.weight.data.copy_(torch.from_numpy(cur_weight))
            else:
                model.encoder.embed_tokens.weight.data.copy_(torch.from_numpy(cur_weight))
            embed_state = state['model']['decoder.embed_tokens.weight'][:len(original_dict), :]
            cur_weight = model.decoder.embed_tokens.weight.detach().numpy()
            cur_weight[:len(original_dict)] = embed_state
            model.decoder.embed_tokens.weight.data.copy_(torch.from_numpy(cur_weight))
        elif getattr(cfg, 'load_finetuned_bart_from', '') != '':
            # load finetuned bart model
            model.decoder = load_pretrained_component_from_model(model.decoder, cfg.load_finetuned_bart_from)
            if isinstance(model.encoder, EncoderWithRec):
                model.encoder.conv_encoder = load_pretrained_component_from_model(
                    model.encoder.conv_encoder, cfg.load_finetuned_bart_from)
            else:
                model.encoder = load_pretrained_component_from_model(model.encoder, cfg.load_finetuned_bart_from)
        elif getattr(cfg, 'load_rec_bart_from', '') != '':
            model.decoder = load_pretrained_component_from_model(model.decoder, cfg.load_rec_bart_from)
            state = load_checkpoint_to_cpu(cfg.load_rec_bart_from)
            component_state_dict = OrderedDict()
            for key in state["model"].keys():
                if key.startswith('encoder.conv_encoder'):
                    component_subkey = key[len('encoder.conv_encoder') + 1:]
                    component_state_dict[component_subkey] = state["model"][key]
            model.encoder.conv_encoder.load_state_dict(component_state_dict, strict=True)
            component_state_dict = OrderedDict()
            for key in state["model"].keys():
                if key.startswith('encoder.recommender.context_to_entity'):
                    component_subkey = key[len('encoder.recommender.context_to_entity') + 1:]
                    component_state_dict[component_subkey] = state["model"][key]
            model.encoder.recommender.context_to_entity.load_state_dict(component_state_dict, strict=True)

        if cfg.share_all_embeddings:
            if isinstance(model.encoder, EncoderWithRec):
                model.decoder.embed_tokens = model.encoder.conv_encoder.embed_tokens
            else:
                model.decoder.embed_tokens = model.encoder.embed_tokens

        if cfg.share_all_embeddings or cfg.share_decoder_input_output_embed:
            model.decoder.output_projection.weight = model.decoder.embed_tokens.weight

        return model

    def forward(self, net_input):
        if isinstance(self.encoder, EncoderWithRec):
            encoder_out = self.encoder(net_input)
        else:
            encoder_out = self.encoder(
                net_input['src_tokens'],
                src_lengths=net_input['src_lengths'],
            )
        x, extra = self.decoder(
            net_input['prev_output_tokens'],
            encoder_out=encoder_out,
            features_only=False,
            src_lengths=net_input['src_lengths'],
        )

        if isinstance(self.encoder, EncoderWithRec):
            return x, extra, encoder_out['movie_scores']
        else:
            return x, extra

    def recommend(self, net_input):
        return self.encoder.recommend(net_input)

    def upgrade_state_dict_named(self, state_dict, name):
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += "."

            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, "upgrade_state_dict_named"):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, "upgrade_state_dict"):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)


@register_model_architecture("convrec_time_bart", "convrec_time_redial")
def base_architecture(args):
    args.encoder_embed_dim = 768
    args.encoder_ffn_embed_dim = 4 * 768
    args.encoder_layers = 6
    args.encoder_attention_heads = 12
    args.decoder_layers = 6
    args.decoder_attention_heads = 12
    args.encoder_normalize_before = False
    args.encoder_learned_pos = True
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = 768
    args.decoder_ffn_embed_dim = 4 * 768
    args.decoder_normalize_before = False
    args.decoder_learned_pos = True
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = True
    args.share_all_embeddings = True

    args.decoder_output_dim = 768
    args.decoder_input_dim = 768

    args.no_scale_embedding = True
    args.layernorm_embedding = True

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)


@register_model_architecture("convrec_time_bart", "convrec_time_opendial")
def convrec_time_opendial(args):
    args.n_entity = 100813
    args.n_relation = 124
    args.n_movie = 8305
    args.relation_count_threshold = 3000
    args.rgcn_n_hop = 1
    args.movieid2entityid_json = None
    args.kg_file = 'kgdata/opendialkg.pkl'

    base_architecture(args)
