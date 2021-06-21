from model.transformer import OurTransformerModel
import jieba

hippop_generator = OurTransformerModel.from_pretrained(
    'checkpoints/transformer_base/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='data/data-bin',
    bpe='subword_nmt',
    bpe_codes='data/src_tgt/code',
)
while True:
    src = input('Input the hip-pop lyric: \n')
    src = list(jieba.cut(src))
    src.reverse()
    tgt = hippop_generator.translate(src).split(' ')
    tgt.reverse()
    print('Continuation:')
    print(''.join(tgt))
    print('\n')
