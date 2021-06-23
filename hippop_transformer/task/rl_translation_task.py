from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
import numpy as np
import torch


@register_task("rl_hippop_translation")
class RLHippopTranslationTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict, rhyme_table=None):
        super(RLHippopTranslationTask, self).__init__(args, src_dict, tgt_dict)
        rhyme_table = np.load('data/rhyme_table.npz')['arr_0']
        self.rhyme_table = torch.tensor(rhyme_table).cuda()