from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from argparse import Namespace
import json
from fairseq.data import (
    encoders,
)
import torch
from ..criterion.gan_loss import get_losses


@register_task("gan_hippop_translation")
class GanHippopTranslationTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super(GanHippopTranslationTask, self).__init__(args, src_dict, tgt_dict)
        self.training_object = 'generator'

    def build_model(self, args):  # TODO discriminator
        model = super().build_model(args)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        # with torch.autograd.profiler.record_function("forward"):
        #     # mle loss
        #     loss, sample_size, logging_output, net_output = criterion(model, sample)
        #     # train discriminator
        #     real_tgts = sample['net_input']['prev_output_tokens']
        #     fake_tgts = torch.argmax(net_output[0], dim=-1)
        #     bert_logits_real = model.discriminator(input_ids=real_tgts, return_dict=True)['logits']
        #     bert_logits_fake = model.discriminator(input_ids=fake_tgts, return_dict=True)['logits']
        #     gen_loss, dis_loss = get_losses(bert_logits_real, bert_logits_fake, model.args.discriminator_bert_loss_type)
        #     # train generator
        with torch.autograd.profiler.record_function("forward"):
            if update_num % 5:  # switch training object
                if self.training_object == 'generator':
                    self.training_object = 'discriminator'
                else:
                    self.training_object = 'generator'

            # gen loss and dis loss
            real_tgts = sample['net_input']['prev_output_tokens']  # [bsz, seq_len]
            fake_tgts = model.forward_generate_gumbel(**sample["net_input"])
            fake_tgts = fake_tgts.contiguous().view(fake_tgts.size(1), fake_tgts.size(0), -1)  # [bsz, seq_len, vocab_size]
            embedding_matrix = model.discriminator.bert.embeddings.word_embeddings.weight
            fake_embs = torch.einsum(
                "ve,bcv -> bce",
                embedding_matrix,
                fake_tgts,
            )
            bert_logits_real = model.discriminator(input_ids=real_tgts, return_dict=True)['logits']
            bert_logits_fake = model.discriminator(inputs_embeds=fake_embs, return_dict=True)['logits']
            gen_loss, dis_loss = get_losses(bert_logits_real, bert_logits_fake, model.args.discriminator_bert_loss_type)

            if self.training_object == 'generator':
                # mle loss
                mle_loss, sample_size, logging_output = criterion(model, sample)
                loss = gen_loss + mle_loss
            else:
                # gp loss(reg loss)
                gp_loss = model.calc_gradient_penalty(real_tgts, fake_tgts)  # gradient penalty
                loss = dis_loss + gp_loss

        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
