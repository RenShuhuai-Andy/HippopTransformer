import numpy as np
from pypinyin import STYLE_FINALS, lazy_pinyin
from collections import defaultdict

word2idx = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
word_list = []
FINALS = 2
# 韵母表 copy from pypinyin
data_for_finals = [
    ['衣', dict(style=FINALS), ['i']],
    ['乌', dict(style=FINALS), ['u']],
    ['迂', dict(style=FINALS), ['v']],
    ['啊', dict(style=FINALS), ['a']],
    ['呀', dict(style=FINALS), ['ia']],
    ['蛙', dict(style=FINALS), ['ua']],
    ['喔', dict(style=FINALS), ['o']],
    ['窝', dict(style=FINALS), ['uo']],
    ['鹅', dict(style=FINALS), ['e']],
    ['耶', dict(style=FINALS), ['ie']],
    ['约', dict(style=FINALS), ['ve']],
    ['哀', dict(style=FINALS), ['ai']],
    ['歪', dict(style=FINALS), ['uai']],
    # ['欸', dict(style=FINALS), ['ei']],
    ['诶', dict(style=FINALS), ['ei']],
    ['威', dict(style=FINALS), ['uei']],
    ['熬', dict(style=FINALS), ['ao']],
    ['腰', dict(style=FINALS), ['iao']],
    ['欧', dict(style=FINALS), ['ou']],
    ['忧', dict(style=FINALS), ['iou']],
    ['安', dict(style=FINALS), ['an']],
    ['烟', dict(style=FINALS), ['ian']],
    ['弯', dict(style=FINALS), ['uan']],
    ['冤', dict(style=FINALS), ['van']],
    ['恩', dict(style=FINALS), ['en']],
    ['因', dict(style=FINALS), ['in']],
    ['温', dict(style=FINALS), ['uen']],
    ['晕', dict(style=FINALS), ['vn']],
    ['昂', dict(style=FINALS), ['ang']],
    ['央', dict(style=FINALS), ['iang']],
    ['汪', dict(style=FINALS), ['uang']],
    ['亨', dict(style=FINALS), ['eng']],
    ['英', dict(style=FINALS), ['ing']],
    ['翁', dict(style=FINALS), ['ueng']],
    ['轰', dict(style=FINALS), ['ong']],
    ['雍', dict(style=FINALS), ['iong']],
    ['儿', dict(style=FINALS), ['er']],
]

with open("dict.tgt.txt", 'r') as f:
    lines = f.readlines()
    for l in lines:
        line, field = l.rsplit(" ", 1)
        word2idx[line] = len(word2idx)
        word_list.append(line)
# rhyme_matrix = np.zeros((len(word2idx), len(word2idx)))
idx2word = {v: k for k, v in word2idx.items()}
ym2word = defaultdict(lambda: list())
ym2wordvec = defaultdict(lambda: np.zeros((len(word2idx))))

ym_table = set([d[2][0] for d in data_for_finals])
print(ym_table)
for i in range(4, len(word2idx)):  # the first 4 token is special token
    word = idx2word[i]
    ym = lazy_pinyin(word, style=STYLE_FINALS)[-1]
    if ym in ym_table:
        ym2word[ym].append(word)
        ym2wordvec[ym][word2idx[word]] = 1

print(len(ym2word))
rhyme_matrix = np.zeros((len(word2idx), len(word2idx)))
for k in ym2word:
    print(ym2word[k][:20])
    print(len(ym2word[k]))
    print(np.sum(ym2wordvec[k]))

for i in range(len(word2idx)):
    word = idx2word[i]
    ym = lazy_pinyin(word, style=STYLE_FINALS)[-1]
    if ym in ym2wordvec:
        rhyme_matrix[word2idx[word]] = ym2wordvec[ym]
    else:
        rhyme_matrix[word2idx[word]] = np.ones((len(word2idx)))  # not found, set to all ones
print(rhyme_matrix.shape)
print(rhyme_matrix[:20, :20])
for i in range(20):
    print(i, idx2word[i])
# np.save("rhyme_table.npy", rhyme_matrix)
np.savez_compressed("rhyme_table.npz", rhyme_matrix)
