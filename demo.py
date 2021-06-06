from model.transformer import OurTransformerModel
hippop_generator = OurTransformerModel.from_pretrained(
  'checkpoints/transformer_base/',
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path='data/data-bin',
  bpe='subword_nmt',
  bpe_codes='data/src_tgt/code'
)
while True:
    src = input('please input the src: \n')
    tgt = hippop_generator.translate(src).split(' ')
    tgt.reverse()
    print('tgt:')
    print(''.join(tgt))