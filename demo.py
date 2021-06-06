from model.transformer import OurTransformerModel
from model.hippop_task import HippopTask
import numpy as np

# load rhyme table
rhyme_table = np.load("data/rhyme_table.npz")['arr_0']

hippop_task = HippopTask()
hippop_generator = OurTransformerModel.from_pretrained(
    'checkpoints/transformer_base/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='data/data-bin',
    bpe='subword_nmt',
    bpe_codes='data/src_tgt/code',
    # task='hippop',
    task=HippopTask,
    rhyme_table=rhyme_table
)
while True:
    src = input('please input the src: \n')
    tgt = hippop_generator.translate(src).split(' ')
    tgt.reverse()
    print('tgt:')
    print(''.join(tgt))
