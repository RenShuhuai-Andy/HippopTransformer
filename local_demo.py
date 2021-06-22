from model.transformer import OurTransformerModel
import jieba
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--checkpoint_dict', type=str, default='checkpoints/transformer_base/')
parse.add_argument('--task', type=str, default='hippop')
args = parse.parse_args()

hippop_generator = OurTransformerModel.from_pretrained(
    args.checkpoint_dict,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='data/data-bin',
    bpe='subword_nmt',
    bpe_codes='data/src_tgt/code',
    task=args.task
)
while True:
    src = input('Input the hip-pop lyric: \n')
    src = list(jieba.cut(src))
    src.reverse()
    tgt = hippop_generator.translate(' '.join(src)).split(' ')
    tgt.reverse()
    print('Continuation:')
    print(''.join(tgt))
    print('\n')
