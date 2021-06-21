from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from model.rhyme_seq_generator import RhymeSequenceGenerator


@register_task("hippop")
class HippopTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict, rhyme_table=None):
        super(HippopTask, self).__init__(args, src_dict, tgt_dict)
        # self.rhyme_table = rhyme_table
        self.rhyme_table = 'data/rhyme_table.npz'

    # modify sequence generator here
    def build_generator(
            self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        return super(HippopTask, self).build_generator(models, args, seq_gen_cls=RhymeSequenceGenerator,
                                                       extra_gen_cls_kwargs={'rhyme_table': self.rhyme_table})
