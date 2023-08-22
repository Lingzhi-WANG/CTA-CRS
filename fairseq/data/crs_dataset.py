import logging

import numpy as np
import torch
import json
from fairseq.data import FairseqDataset, LanguagePairDataset, data_utils

logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths,},
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    return batch, sort_order


class CRSDataset(LanguagePairDataset):
    def __init__(
            self,
            src,
            src_sizes,
            src_dict,
            tgt=None,
            tgt_sizes=None,
            tgt_dict=None,
            left_pad_source=True,
            left_pad_target=False,
            shuffle=True,
            input_feeding=True,
            remove_eos_from_source=False,
            append_eos_to_target=False,
            align_dataset=None,
            constraints=None,
            append_bos=False,
            eos=None,
            num_buckets=0,
            src_lang_id=None,
            tgt_lang_id=None,
            pad_to_multiple=1,
            movie_id_range=(50263, 56900),  # 56899 50263
            movie_cate_json=None,
            movie_neighbor_json=None,
            movie2entity_json=None,
            no_movie_idx=False,
    ):
        super().__init__(
            src,
            src_sizes,
            src_dict,
            tgt,
            tgt_sizes,
            tgt_dict,
            left_pad_source,
            left_pad_target,
            shuffle,
            input_feeding,
            remove_eos_from_source,
            append_eos_to_target,
            align_dataset,
            constraints,
            append_bos,
            eos,
            num_buckets,
            src_lang_id,
            tgt_lang_id,
            pad_to_multiple,
        )
        self.movie_id_range = movie_id_range
        self.movie_num = movie_id_range[1] - movie_id_range[0]
        self.no_movie_idx = no_movie_idx
        if movie_cate_json is not None:
            with open(movie_cate_json, 'r') as f:
                cate_dict = json.load(f)
                self.cate_dict = {src_dict.index(x): np.array(cate_dict[x]) for x in cate_dict.keys()}
                self.cate_num = len(self.cate_dict[list(self.cate_dict.keys())[0]])
        else:
            self.cate_dict = None
        if movie_neighbor_json is not None:
            with open(movie_neighbor_json, 'r') as f:
                neighbor_dict = json.load(f)
                self.neighbor_dict = {src_dict.index(x): neighbor_dict[x] for x in neighbor_dict.keys()}
                self.neighbor_num = 5832
        else:
            self.neighbor_dict = None
        if movie2entity_json is not None:
            with open(movie2entity_json, 'r') as f:
                self.movieid2entityid = json.load(f)
        else:
            self.movieid2entityid = None

    def __getitem__(self, index):
        tgt_item = self.tgt[index]
        src_item = self.src[index]

        sep_idx = self.src_dict.index('[SEP]')
        is_sep = src_item == sep_idx
        sep_indices = torch.nonzero(is_sep, as_tuple=False).squeeze(-1)
        assert len(sep_indices) == 3
        mv_cates = [self.src_dict[src_item[x]] for x in range(int(sep_indices[0])+1, int(sep_indices[1]))]
        mv_cates = torch.LongTensor([int(c[1:]) for c in mv_cates])
        wd_cates = [self.src_dict[src_item[x]] for x in range(int(sep_indices[1])+1, int(sep_indices[2]))]
        wd_cates = torch.LongTensor([int(c[1:]) for c in wd_cates])
        text_entities = [self.src_dict[src_item[x]] for x in range(int(sep_indices[2])+1, len(src_item)-1)]
        if self.movieid2entityid is not None:
            final_text_entities = []
            for e in text_entities:
                if e[0] == 'E':
                    final_text_entities.append(int(e[1:]))
                elif e in self.movieid2entityid:
                    final_text_entities.append(self.movieid2entityid[e])
                else:
                    final_text_entities.append(-1)
            text_entities = torch.LongTensor(final_text_entities)
        else:
            text_entities = torch.LongTensor([int(e[1:]) for e in text_entities])
        src_item = torch.cat([src_item[:int(sep_indices[0])], src_item[-1:]])

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            "movie_cates": mv_cates,
            "word_cates": wd_cates,
            "text_entities": text_entities,
        }

        return example

    def collater(self, samples, pad_to_length=None):
        res, sort_order = collate(samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(), pad_to_length=pad_to_length)
        src_tokens = res['net_input']['src_tokens']  # B x T
        target = res['target']  # B x T
        bsz = src_tokens.size(0)

        if not self.no_movie_idx:
            mv_cates = []
            wd_cates = []
            text_entities = []
            for b in sort_order:
                mv_cates.append(samples[int(b)]["movie_cates"])
                wd_cates.append(samples[int(b)]["word_cates"])
                text_entities.append(samples[int(b)]["text_entities"])
            res["net_input"]["movie_cates"] = mv_cates
            res["net_input"]["word_cates"] = wd_cates
            res["net_input"]["text_entities"] = text_entities

        movie_sets = torch.zeros((bsz, self.movie_num), dtype=torch.float32).to(src_tokens.device)  # B x M
        is_movie = (src_tokens >= self.movie_id_range[0]) & (src_tokens < self.movie_id_range[1])  # B x T
        movie_idxs = []
        for b in range(bsz):
            idxs = torch.nonzero(is_movie[b], as_tuple=False).squeeze(-1)
            mvs = torch.index_select(src_tokens[b], dim=0, index=idxs)
            movie_sets[b][mvs - self.movie_id_range[0]] = 1
            movie_idxs.append(mvs - self.movie_id_range[0])

        target_movie_sets = torch.zeros((bsz, self.movie_num), dtype=torch.float32).to(target.device)  # B x M
        is_movie = (target >= self.movie_id_range[0]) & (target < self.movie_id_range[1])  # B x T
        if self.cate_dict is not None:
            movie_cate_sets = torch.zeros((bsz, self.cate_num), dtype=torch.float32).to(src_tokens.device)
        else:
            movie_cate_sets = None
        if self.neighbor_dict is not None:
            movie_neighbor_sets = torch.zeros((bsz, self.neighbor_num), dtype=torch.float32).to(src_tokens.device)
        else:
            movie_neighbor_sets = None
        for b in range(bsz):
            idxs = torch.nonzero(is_movie[b], as_tuple=False).squeeze(-1)
            mvs = torch.index_select(target[b], dim=0, index=idxs)
            if len(mvs) > 0:
                if self.cate_dict is not None:
                    movie_cate_sets[b] = torch.from_numpy(
                        np.sum([self.cate_dict[int(mv)] for mv in mvs], axis=0)).to(movie_cate_sets)
                if self.neighbor_dict is not None:
                    neighbor_idxs = torch.from_numpy(
                        np.concatenate([self.neighbor_dict[int(mv)] for mv in mvs])).to(target)
                    movie_neighbor_sets[b][neighbor_idxs] = 1
            target_movie_sets[b][mvs - self.movie_id_range[0]] = 1

        if self.neighbor_dict is not None and self.cate_dict is not None:
            movie_neighbor_sets = torch.cat([movie_neighbor_sets, movie_cate_sets], dim=-1)
            movie_neighbor_sets = torch.where(movie_neighbor_sets > 1, torch.ones_like(movie_neighbor_sets), movie_neighbor_sets)

        res['net_input']['movie_sets'] = movie_sets
        if not self.no_movie_idx:
            res['net_input']['movie_idxs'] = movie_idxs
        res['target_movie_sets'] = target_movie_sets
        if movie_cate_sets is not None:
            res['net_input']['movie_cate_sets'] = movie_cate_sets
        if movie_neighbor_sets is not None:
            res['net_input']['movie_neighbor_sets'] = movie_neighbor_sets

        return res

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        return indices


