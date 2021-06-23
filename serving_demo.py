#!/usr/bin/python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template
from flask_wtf import FlaskForm
from flask_pagedown import PageDown
from flask_pagedown.fields import PageDownField
from wtforms.fields import SubmitField
import jieba 
from hippop_transformer.model.transformer import OurTransformerModel


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
pagedown = PageDown(app)

print('loading generator')
hippop_generator = OurTransformerModel.from_pretrained(
    '/home/renshuhuai/Chinese-Hippop-Generation/checkpoints/transformer_base/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/home/renshuhuai/Chinese-Hippop-Generation/data/data-bin',
    bpe='subword_nmt',
    bpe_codes='/home/renshuhuai/Chinese-Hippop-Generation/data/src_tgt/code',
    task='hippop'
)
print('loading finished')


class PageDownFormExample(FlaskForm):
    pagedown = PageDownField('进行生成')
    submit = SubmitField('开始创作!')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = PageDownFormExample()
    text = None
    if form.validate_on_submit():
        source = form.pagedown.data #.lower()
        segs = list(jieba.cut(source))
        segs.reverse()
        text = " ".join(segs)
        text = hippop_generator.translate(' '.join(segs)).split(' ')
        text.reverse()
        text = "".join(text)
    else:
        form.pagedown.data = ('测试一下分词的效果')
    return render_template('index.html', form=form, text=text)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006)

if __name__ == '__main__':
    app.run(debug=True)
