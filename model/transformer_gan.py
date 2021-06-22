# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
sys.path.append("utils")
from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    AdamW,
    BertForSequenceClassification
)
from fairseq.models.transformer import TransformerModel, base_architecture
from model.transformer import transformer_base_architecture
from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)

dis_filter_sizes = [2, 3, 4, 5]
dis_num_filters = [300, 300, 300, 300]
DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


# dis_filter_sizes = [5]
# dis_num_filters = [300]

@register_model("transformer_gan")
class TransformerGAN(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.temperature = 1

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # args discriminator
        parser.add_argument('--discriminator_type', default='bert', type=str)
        parser.add_argument('--discriminator_bert_model_path', default='checkpoints/bert-base-uncased/', type=str)
        parser.add_argument('--discriminator_bert_loss_type', default='wgan-gp', type=str)
        parser.add_argument('--discriminator_bert_model_type', default='bert', type=str)
        parser.add_argument('--discriminator_bert_random_weights', default=None)
        parser.add_argument('--discriminator_bert_freeze_layers', default=['0', '1', '2', '3', '4'])
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        cls.build_discriminator(args, task)
        return cls.build_generator(args, task)

    @classmethod
    def build_generator(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        transformer_base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        cls.generator = cls(args, encoder, decoder)
        return cls.generator

    @classmethod
    def build_discriminator(cls, args, task):
        """Build a new model instance."""

        # Create discriminator
        if args.discriminator_type == "bert":
            # Can change d_embed
            cls.discriminator = cls.create_bert_model(
                args.discriminator_bert_model_path, args.discriminator_bert_loss_type,
                args.discriminator_bert_model_type, args.discriminator_bert_random_weights
            )
            cls.discriminator.unfreeze_idx = cls.calculate_unfreeze_idx(args)
        else:
            cls.discriminator = None
        cls.discriminator.cuda()
        # return cls.discriminator

    @classmethod
    def create_bert_model(cls, model_name_or_path, loss_type, model_type=None, random_weights=False):

        config_class = BertConfig
        config = config_class.from_pretrained(model_name_or_path, cache_dir=None)

        if model_type == "bert_lm":
            if random_weights:
                print("Starting from random")
                model = BertForSequenceClassification(config=config)
            else:
                model_class = BertForMaskedLM
                model_lm = model_class.from_pretrained(
                    model_name_or_path,
                    from_tf=bool(".ckpt" in model_name_or_path),
                    config=config,
                    cache_dir=None,
                )
                model = BertForSequenceClassification(config=config)
                model.bert = model_lm.bert

        else:
            if random_weights:
                raise NotImplementedError
            model_class = BertForSequenceClassification
            model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
                cache_dir=None,
            )

        return model.bert if loss_type == "mmd" else model

    @classmethod
    def calculate_unfreeze_idx(cls, args):
        cn, unfreeze_idx, layers = 0, [], []
        for name, param in cls.discriminator.named_parameters():
            if name.startswith("bert.embeddings") and not args.discriminator_bert_random_weights:
                pass
            elif name.startswith("bert.encoder.layer") and name.split('.')[3] in args.discriminator_bert_freeze_layers:
                pass
            else:
                unfreeze_idx.append(cn)
            cn += 1

            if name.startswith("bert.encoder.layer"):
                layers.append(name.split('.')[3])

        # check the total number of layers in the BERT >= the number of layers to be freeze
        assert len(layers) >= len(args.discriminator_bert_freeze_layers)

        return unfreeze_idx

    def forward_generate_gumbel(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = False,
            features_only: bool = True,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            temperature=1):  # TODO

        from torch.autograd import Variable

        # if data.device.index == 0 :
        # print(data.device, data.shape, data[0].nonzero())

        def sample_gumbel(shape, eps=1e-20):
            U = torch.rand(shape).cuda()
            return -Variable(torch.log(-torch.log(U + eps) + eps))

        def gumbel_softmax_sample(logits, temperature):
            y = logits + sample_gumbel(logits.size())
            return F.softmax(y / temperature, dim=-1)

        def gumbel_softmax(logits, temperature):
            """
            input: [*, n_class]
            return: [*, n_class] an one-hot vector
            """
            y = gumbel_softmax_sample(logits, temperature)
            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            return (y_hard - y).detach() + y

        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        logits = decoder_out[0]
        tgt_len = logits.size(1)
        batch_size = logits.size(0)
        logits = gumbel_softmax(
            logits.contiguous().view(tgt_len, batch_size, -1), temperature=temperature
        )

        return logits

    def calc_gradient_penalty(self, real_data, fake_data, LAMBDA=10):
        alpha = torch.rand([real_data.shape[0], 1, 1], device=real_data.device)
        alpha = alpha.expand(real_data.size())

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        interpolates = torch.einsum(
            "ve,bcv -> bce",
            self.discriminator.bert.embeddings.word_embeddings.weight,
            interpolates,
        )
        disc_interpolates = self.discriminator(inputs_embeds=interpolates)[0][:, 0]

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size(),
                                                                device=real_data.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(real_data.shape[0], -1)

        # https://github.com/igul222/improved_wgan_training/blob/master/gan_language.py
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((slopes - 1.) ** 2).mean() * LAMBDA

        return gradient_penalty


@register_model_architecture("transformer_gan", "transformer_gan_base")
def transformer_gan(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    transformer_base_architecture(args)
