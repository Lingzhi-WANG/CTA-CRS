# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import numpy as np
import collections


@dataclass
class CRSLossConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    generate_ratio: float = field(
        default=1.0,
        metadata={"help": "training ratio for generation"},
    )
    recommend_ratio: float = field(
        default=0.0,
        metadata={"help": "training ratio for recommendation"},
    )
    bart_recommend_ratio: float = field(
        default=0.0,
        metadata={"help": "training ratio for recommendation with bart reps"},
    )
    bart_mlp_l1_weight: float = field(
        default=0.0,
        metadata={"help": "l1 regularization for mlp layer of bart recommendation"},
    )
    report_recall: bool = field(
        default=False,
        metadata={"help": "report recall metrics"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("crs_time_loss", dataclass=CRSLossConfig)
class CRSTimeLoss(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        generate_ratio=1.0,
        recommend_ratio=0.0,
        bart_recommend_ratio=0.0,
        report_recall=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.generate_ratio = generate_ratio
        self.recommend_ratio = recommend_ratio
        self.bart_recommend_ratio = bart_recommend_ratio
        self.report_recall = report_recall

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(sample["net_input"])
        if self.generate_ratio > 0:
            generate_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        else:
            generate_loss, nll_loss = 0, 0
        if self.recommend_ratio > 0:
            movie_net_output = net_output[2][0]
            assert movie_net_output is not None
            # only consider samples that have movie items
            has_target = torch.nonzero(sample['target_movie_sets'], as_tuple=False)
            batch_idxs = has_target[:, 0]
            targets = has_target[:, 1]
            entity_ids = model.encoder.idx2entityid[targets]
            recommend_loss = F.cross_entropy(
                movie_net_output.index_select(dim=0, index=batch_idxs), entity_ids, ignore_index=-1, reduction='sum')
        else:
            recommend_loss = 0
        if self.bart_recommend_ratio > 0:
            assert isinstance(net_output[2], tuple)
            bart_net_output = net_output[2][1]
            assert bart_net_output is not None
            # only consider samples that have movie items
            has_target = torch.nonzero(sample['target_movie_sets'], as_tuple=False)
            batch_idxs = has_target[:, 0]
            targets = has_target[:, 1]
            entity_ids = model.encoder.idx2entityid[targets]
            bart_recommend_loss = F.cross_entropy(
                bart_net_output.index_select(dim=0, index=batch_idxs), entity_ids, ignore_index=-1, reduction='sum')
        else:
            bart_recommend_loss = 0

        loss = self.generate_ratio * generate_loss + \
               self.recommend_ratio * recommend_loss + self.bart_recommend_ratio * bart_recommend_loss

        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data if nll_loss != 0 else 0,
            "rec_loss": recommend_loss.data if recommend_loss != 0 else 0,
            "bart_rec_loss": bart_recommend_loss.data if bart_recommend_loss != 0 else 0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        if self.report_recall:
            rec1, rec50, bart_rec1, bart_rec50, total_num = 0.0, 0.0, 0.0, 0.0, 0.0
            if self.recommend_ratio > 0:
                movie_rec_out = F.softmax(movie_net_output, dim=1)
            else:
                movie_rec_out = None
            if self.bart_recommend_ratio > 0:
                bart_rec_out = F.softmax(bart_net_output, dim=1)
            else:
                bart_rec_out = None
            for i, sample_id in enumerate(sample["id"].tolist()):
                tgt = sample['target_movie_sets'][i]
                ground_truths = model.idx2entityid[torch.nonzero(tgt, as_tuple=False).squeeze(-1)]
                if movie_rec_out is not None:
                    movie_res = movie_rec_out[i]
                    movie_res = torch.where(model.entity_is_movie == 1, movie_res, torch.zeros_like(movie_res))
                    _, movie_idxs = torch.topk(movie_res, 50, sorted=True)
                else:
                    movie_idxs = None
                if bart_rec_out is not None:
                    bart_res = bart_rec_out[i]
                    bart_res = torch.where(model.entity_is_movie == 1, bart_res, torch.zeros_like(bart_res))
                    _, bart_idxs = torch.topk(bart_res, 50, sorted=True)
                else:
                    bart_idxs = None
                for gt in ground_truths:
                    if gt == -1:
                        continue
                    if movie_idxs is not None and gt == movie_idxs[0]:
                        rec1 += 1
                    if movie_idxs is not None and gt in movie_idxs:
                        rec50 += 1
                    if bart_idxs is not None and gt == bart_idxs[0]:
                        bart_rec1 += 1
                    if bart_idxs is not None and gt in bart_idxs:
                        bart_rec50 += 1
                    total_num += 1
            if movie_rec_out is None:
                logging_output['rec1'] = bart_rec1
                logging_output['rec50'] = bart_rec50
            else:
                logging_output['rec1'] = rec1
                logging_output['rec50'] = rec50
            if bart_rec_out is not None:
                logging_output['bart_rec1'] = bart_rec1
                logging_output['bart_rec50'] = bart_rec50
            logging_output['total_num'] = total_num

        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        rec_loss_sum = sum(log.get("rec_loss", 0) for log in logging_outputs)
        bart_rec_loss_sum = sum(log.get("bart_rec_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        if rec_loss_sum > 0:
            metrics.log_scalar(
                "rec_loss", rec_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        if bart_rec_loss_sum > 0:
            metrics.log_scalar(
                "bart_rec_loss", bart_rec_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

        total_num = sum(log.get("total_num", 0) for log in logging_outputs)
        if total_num > 0:
            rec1 = sum(log.get("rec1", 0) for log in logging_outputs) / total_num
            rec50 = sum(log.get("rec50", 0) for log in logging_outputs) / total_num
            metrics.log_scalar("rec1", rec1, round=3)
            metrics.log_scalar("rec50", rec50, round=3)
            metrics.log_scalar("rec", rec1+rec50, round=3)
            bart_rec1 = sum(log.get("bart_rec1", 0) for log in logging_outputs) / total_num
            bart_rec50 = sum(log.get("bart_rec50", 0) for log in logging_outputs) / total_num
            if bart_rec50 > 0:
                metrics.log_scalar("bart_rec1", bart_rec1, round=3)
                metrics.log_scalar("bart_rec50", bart_rec50, round=3)
                metrics.log_scalar("both_rec50", rec50 + bart_rec50, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
