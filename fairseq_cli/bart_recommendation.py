#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
import collections
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "recommendation-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    # # Set dictionaries
    # try:
    #     src_dict = getattr(task, "source_dictionary", None)
    # except NotImplementedError:
    #     src_dict = None
    # tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )
    # assert len(models) == 1

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)
    task.load_dataset(cfg.dataset.valid_subset, task_cfg=saved_cfg.task)

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    if cfg.model.recommend_only_one_hop:
        n_entity = models[0].encoder.recommender.n_entity
        one_hop_entities = collections.defaultdict(list)
        for entities in models[0].encoder.recommender.db_edge_idx.t():
            assert len(entities) == 2
            one_hop_entities[int(entities[0])].append(int(entities[1]))
            one_hop_entities[int(entities[1])].append(int(entities[0]))
        for entity in one_hop_entities.keys():
            one_hop_entities[entity] = torch.LongTensor(list(set(one_hop_entities[entity])))
            if use_cuda:
                one_hop_entities[entity] = one_hop_entities[entity].cuda()


    bart_score_weight = 0.0
    best_bart_score_weight = -1
    best_recall = -1
    while bart_score_weight <= 1.0:
        models[0].encoder.recommender.context_rep_weight = bart_score_weight

        # Load valid dataset (possibly sharded)
        itr = task.get_batch_iterator(
            dataset=task.dataset(cfg.dataset.valid_subset),
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(), *[m.max_positions() for m in models]
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=cfg.distributed_training.distributed_world_size,
            shard_id=cfg.distributed_training.distributed_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

        gen_timer = StopwatchMeter()
        num_sentences = 0
        wps_meter = TimeMeter()
        if cfg.model.recommend_only_one_hop:
            rec1, rec3, rec5, rec10, rec25 = 0.0, 0.0, 0.0, 0.0, 0.0
            total1, total3, total5, total10, total25 = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            rec1, rec10, rec50, total_num = 0.0, 0.0, 0.0, 0.0
        for sample in progress:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if "net_input" not in sample:
                continue

            gen_timer.start()
            if len(models) == 1:
                rec_out = models[0].recommend(sample["net_input"])
            else:
                rec_out = []
                for model in models:
                    rec_out.append(model.recommend(sample["net_input"]))
                rec_out = torch.stack(rec_out, dim=0)
                rec_out = rec_out.sum(dim=0)
            gen_timer.stop(len(rec_out))

            for i, sample_id in enumerate(sample["id"].tolist()):
                assert sample.get('target_movie_sets', None) is not None

                res = rec_out[i]
                tgt = sample['target_movie_sets'][i]

                if cfg.model.recommend_only_one_hop:
                    movie_idxs = sample['net_input']['movie_idxs'][i]
                    if len(movie_idxs) > 0:
                        last_entity = models[0].idx2entityid[movie_idxs[-1]]
                        if last_entity != -1:
                            one_hop = torch.zeros(n_entity).to(res)
                            one_hop.index_fill_(-1, one_hop_entities[int(last_entity)], 1)
                            res = torch.where(one_hop == 1, res, torch.zeros_like(res).fill_(-1))

                if getattr(models[0], 'entity_output', False):
                    assert cfg.model.ignore_movies_not_in_kg and getattr(models[0], 'idx2entityid', None) is not None
                    ground_truths = models[0].idx2entityid[torch.nonzero(tgt, as_tuple=False).squeeze(-1)]
                    res = torch.where(models[0].entity_is_movie == 1, res, torch.zeros_like(res))
                else:
                    ground_truths = torch.nonzero(tgt, as_tuple=False).squeeze(-1)
                _, idxs = torch.topk(res, 50, sorted=True)
                for gt in ground_truths:
                    if cfg.model.ignore_movies_not_in_kg:  # do not consider the movies not in KG
                        assert getattr(models[0], 'idx2entityid', None) is not None
                        if getattr(models[0], 'entity_output', False):
                            if gt == -1:
                                continue
                        elif models[0].idx2entityid[gt] == -1:
                            continue
                    if cfg.model.recommend_only_one_hop:
                        if len(movie_idxs) == 0 or len(one_hop_entities[int(last_entity)]) >= 1:
                            if gt == idxs[0]:
                                rec1 += 1
                            total1 += 1
                        if len(movie_idxs) == 0 or len(one_hop_entities[int(last_entity)]) >= 3:
                            if gt in idxs[:3]:
                                rec3 += 1
                            total3 += 1
                        if len(movie_idxs) == 0 or len(one_hop_entities[int(last_entity)]) >= 5:
                            if gt in idxs[:5]:
                                rec5 += 1
                            total5 += 1
                        if len(movie_idxs) == 0 or len(one_hop_entities[int(last_entity)]) >= 10:
                            if gt in idxs[:10]:
                                rec10 += 1
                            total10 += 1
                        if len(movie_idxs) == 0 or len(one_hop_entities[int(last_entity)]) >= 25:
                            if gt in idxs[:25]:
                                rec25 += 1
                            total25 += 1
                    else:
                        if gt == idxs[0]:
                            rec1 += 1
                            rec10 += 1
                            rec50 += 1
                        elif gt in idxs[:10]:
                            rec10 += 1
                            rec50 += 1
                        elif gt in idxs:
                            rec50 += 1
                        total_num += 1

            wps_meter.update(len(rec_out))
            num_sentences += sample["id"].numel()

        if cfg.model.recommend_only_one_hop:
            rec1 /= total1
            rec3 /= total3
            rec5 /= total5
            rec10 /= total10
            rec25 /= total25
            print(
                "Recommend {} results (bart score weight: {}): Recall@1: {}, Recall@3: {}, Recall@5: {}, Recall@10: {}, Recall@25: {}".format(
                    cfg.dataset.valid_subset, bart_score_weight, rec1, rec3, rec5, rec10, rec25
                ),
                file=output_file,
            )
        else:
            rec1 /= total_num
            rec10 /= total_num
            rec50 /= total_num
            print(
                "Recommend {} results (bart score weight: {}): Recall@1: {}, Recall@10: {}, Recall@50: {}".format(
                    cfg.dataset.valid_subset, bart_score_weight, rec1, rec10, rec50
                ),
                file=output_file,
        )
        if cfg.model.recommend_only_one_hop:
            if rec1 + rec25 > best_recall:
                best_recall = rec1 + rec25
                best_bart_score_weight = bart_score_weight
        else:
            if rec1 + rec50 > best_recall:
                best_recall = rec1 + rec50
                best_bart_score_weight = bart_score_weight

        bart_score_weight += 0.1

    models[0].encoder.recommender.context_rep_weight = best_bart_score_weight

    # Load test dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    gen_timer = StopwatchMeter()

    num_sentences = 0
    wps_meter = TimeMeter()
    if cfg.model.recommend_only_one_hop:
        rec1, rec3, rec5, rec10, rec25 = 0.0, 0.0, 0.0, 0.0, 0.0
        total1, total3, total5, total10, total25 = 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        rec1, rec10, rec50, total_num = 0.0, 0.0, 0.0, 0.0
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        gen_timer.start()
        if len(models) == 1:
            rec_out = models[0].recommend(sample["net_input"])
        else:
            rec_out = []
            for model in models:
                rec_out.append(model.recommend(sample["net_input"]))
            rec_out = torch.stack(rec_out, dim=0)
            rec_out = rec_out.sum(dim=0)
        gen_timer.stop(len(rec_out))

        for i, sample_id in enumerate(sample["id"].tolist()):
            assert sample.get('target_movie_sets', None) is not None

            res = rec_out[i]
            tgt = sample['target_movie_sets'][i]

            if cfg.model.recommend_only_one_hop:
                movie_idxs = sample['net_input']['movie_idxs'][i]
                if len(movie_idxs) > 0:
                    last_entity = models[0].idx2entityid[movie_idxs[-1]]
                    one_hop = torch.zeros(n_entity).to(res)
                    one_hop.index_fill_(-1, one_hop_entities[int(last_entity)], 1)
                    res = torch.where(one_hop == 1, res, torch.zeros_like(res).fill_(-1))

            if getattr(models[0], 'entity_output', False):
                assert cfg.model.ignore_movies_not_in_kg and getattr(models[0], 'idx2entityid', None) is not None
                ground_truths = models[0].idx2entityid[torch.nonzero(tgt, as_tuple=False).squeeze(-1)]
                res = torch.where(models[0].entity_is_movie == 1, res, torch.zeros_like(res))
            else:
                ground_truths = torch.nonzero(tgt, as_tuple=False).squeeze(-1)
            _, idxs = torch.topk(res, 50, sorted=True)
            for gt in ground_truths:
                if cfg.model.ignore_movies_not_in_kg:  # do not consider the movies not in KG
                    assert getattr(models[0], 'idx2entityid', None) is not None
                    if getattr(models[0], 'entity_output', False):
                        if gt == -1:
                            continue
                    elif models[0].idx2entityid[gt] == -1:
                        continue
                if cfg.model.recommend_only_one_hop:
                    if len(movie_idxs) == 0 or len(one_hop_entities[int(last_entity)]) >= 1:
                        if gt == idxs[0]:
                            rec1 += 1
                        total1 += 1
                    if len(movie_idxs) == 0 or len(one_hop_entities[int(last_entity)]) >= 3:
                        if gt in idxs[:3]:
                            rec3 += 1
                        total3 += 1
                    if len(movie_idxs) == 0 or len(one_hop_entities[int(last_entity)]) >= 5:
                        if gt in idxs[:5]:
                            rec5 += 1
                        total5 += 1
                    if len(movie_idxs) == 0 or len(one_hop_entities[int(last_entity)]) >= 10:
                        if gt in idxs[:10]:
                            rec10 += 1
                        total10 += 1
                    if len(movie_idxs) == 0 or len(one_hop_entities[int(last_entity)]) >= 25:
                        if gt in idxs[:25]:
                            rec25 += 1
                        total25 += 1
                else:
                    if gt == idxs[0]:
                        rec1 += 1
                        rec10 += 1
                        rec50 += 1
                    elif gt in idxs[:10]:
                        rec10 += 1
                        rec50 += 1
                    elif gt in idxs:
                        rec50 += 1
                    total_num += 1

        wps_meter.update(len(rec_out))
        num_sentences += sample["id"].numel()

    if cfg.model.recommend_only_one_hop:
        rec1 /= total1
        rec3 /= total3
        rec5 /= total5
        rec10 /= total10
        rec25 /= total25
    else:
        rec1 /= total_num
        rec10 /= total_num
        rec50 /= total_num

    logger.info(
        "Recommended {:,} sentences ({:,} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    if cfg.model.recommend_only_one_hop:
        print(
            "Recommend {} results (bart score weight: {}): Recall@1: {}, Recall@3: {}, Recall@5: {}, Recall@10: {}, Recall@25: {}".format(
                cfg.dataset.gen_subset, best_bart_score_weight, rec1, rec3, rec5, rec10, rec25
            ),
            file=output_file,
        )
        return rec1, rec3, rec5, rec10, rec25
    else:
        print(
            "Recommend {} results (bart score weight: {}): Recall@1: {}, Recall@10: {}, Recall@50: {}".format(
                cfg.dataset.gen_subset, best_bart_score_weight, rec1, rec10, rec50
            ),
            file=output_file,
        )
        return rec1, rec10, rec50


def cli_main():
    parser = options.get_generation_parser()
    # TODO: replace this workaround with refactoring of `AudioPretraining`
    parser.add_argument(
        '--arch', '-a', metavar='ARCH', default="transformer",
        help='Model architecture. For constructing tasks that rely on '
             'model args (e.g. `AudioPretraining`)'
    )
    parser.add_argument(
        '--ignore-movies-not-in-kg', action='store_true',
    )
    parser.add_argument(
        '--recommend-only-one-hop', action='store_true',
    )
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
