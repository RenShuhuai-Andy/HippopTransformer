from fairseq.models import register_model_architecture
from .transformer import transformer_base_architecture
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from .rhyme_seq_generator import RhymeSequenceGenerator
import numpy as np


@register_model_architecture('our_transformer', 'transformer_base')
def transformer_base(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    transformer_base_architecture(args)


@register_model_architecture('our_transformer', 'transformer_large')
def transformer_large(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    transformer_base_architecture(args)


@register_task("hippop")
class HippopTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict, rhyme_table=None):
        super(HippopTask, self).__init__(args, src_dict, tgt_dict)
        self.rhyme_table = np.load('data/rhyme_table.npz')['arr_0']
        # self.rhyme_table = rhyme_table

    # modify sequence generator here
    def build_generator(
            self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        return super(HippopTask, self).build_generator(models, args, seq_gen_cls=RhymeSequenceGenerator,
                                                       extra_gen_cls_kwargs={'rhyme_table': self.rhyme_table})
