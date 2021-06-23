import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterionConfig
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss, LabelSmoothedCrossEntropyCriterion
from fairseq import metrics, utils
import math


# Util function for gumbel softmax
def sample_gumbel(logits, eps=1e-20):
    U = torch.rand_like(logits)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits)
    return F.softmax(y / temperature, dim=-1)  # bsz, seq_len, vocab_size


def gumbel_softmax(logits, temperature):
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()  # bsz, seq_len, vocab_size
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard  # bsz, seq_len, vocab_size,  one-hot vector


@register_criterion("rhyme_cross_entropy", dataclass=CrossEntropyCriterionConfig)
class RhymeCrossEntropyCriterion(FairseqCriterion):
    """Use this to replace the original CE criterion for incorporating sentence-level reward for rhymed target"""

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        rhyme_loss, ce_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        loss = rhyme_loss + ce_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "rhyme_loss": rhyme_loss.data,
            "ce_loss": ce_loss.data,
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, temperature=5.0, rhyme_weight=1.0):
        # Get one-hot output using Gumbel-softmax for passing the gradient
        logits = net_output[0]
        y_hard = gumbel_softmax(logits, temperature=temperature)

        # Rhyme Loss, 1 if the generated sentences is rhymed otherwise 0
        predicted_first_token = y_hard[:, 0, :]  # bsz, vocab_size
        source_first_token = sample['net_input']['src_tokens'][:, 0]  # bsz

        # Get rhyme table from the task
        rhyme_table = self.task.rhyme_table
        rhyme_indicator = torch.index_select(rhyme_table, 0, source_first_token)  # bsz , vocab_size
        rhyme_score = torch.sum(predicted_first_token * rhyme_indicator.int(), dim=-1).sum()  # (,)
        rhyme_loss = - (rhyme_weight * rhyme_score)
        # Normal CE Loss
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        ce_loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return rhyme_loss, ce_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        rhyme_loss_sum = sum(log.get("rhyme_loss", 0) for log in logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "rhyme_loss", rhyme_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ce_loss", ce_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )


@register_criterion("rhyme_label_smoothed_cross_entropy")
class RhymeLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    """Use this to replace the original CE criterion for incorporating sentence-level reward for rhymed target"""

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        rhyme_loss, ce_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        loss = rhyme_loss + ce_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "rhyme_loss": rhyme_loss.data,
            "ce_loss": ce_loss.data,
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, temperature=5.0, rhyme_weight=1.0):
        # Get one-hot output using Gumbel-softmax for passing the gradient
        logits = net_output[0]
        y_hard = gumbel_softmax(logits, temperature=temperature)

        # Rhyme Loss, 1 if the generated sentences is rhymed otherwise 0
        predicted_first_token = y_hard[:, 0, :]  # bsz, vocab_size
        source_first_token = sample['net_input']['src_tokens'][:, 0]  # bsz

        # Get rhyme table from the task
        rhyme_table = self.task.rhyme_table
        rhyme_indicator = torch.index_select(rhyme_table, 0, source_first_token)  # bsz , vocab_size
        rhyme_score = torch.sum(predicted_first_token * rhyme_indicator.int(), dim=-1).sum()  # (,)
        rhyme_loss = - (rhyme_weight * rhyme_score)

        # Normal CE Loss
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        ce_loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return rhyme_loss, ce_loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        rhyme_loss_sum = sum(log.get("rhyme_loss", 0) for log in logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "rhyme_loss", rhyme_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ce_loss", ce_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
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