import os
import pandas as pd
import random


def data_preprocess():
    lyric_list = []
    path = os.listdir('data/origin_lyric')
    for file in path:
        try:
            df = pd.read_csv(os.path.join('data/origin_lyric', file), encoding='gbk', header=None)
        except:
            try:
                df = pd.read_excel(os.path.join('data/origin_lyric', file), header=None)
            except:
                df = pd.read_csv(os.path.join('data/origin_lyric', file), encoding='gb18030', header=None)
        df = df[0].dropna(axis=0, how='any')
        current_lyric = df.tolist()
        assert all(isinstance(x, str) for x in current_lyric)
        lyric_list.extend(current_lyric)

    single_len = len(lyric_list)
    train_pair_len = int(single_len * 0.9)
    dev_pair_len = int(single_len * 0.05)
    test_pair_len = int(single_len * 0.05)
    print('train num: %d, dev num: %d, test num: %d' % (train_pair_len, dev_pair_len, test_pair_len))

    lyric_list = [[lyric_list[i], lyric_list[i+1]] for i in range(0, single_len-1, 2)]
    random.shuffle(lyric_list)
    lyric_list = sum(lyric_list, [])

    pair = []
    with open(r'data/dataset/total_train.txt', 'w', encoding='utf-8') as f:
        for lyric in lyric_list[:train_pair_len]:
            pair.append(lyric)
            if len(pair) == 2:
                f.write(pair[0] + '， ' + pair[1] + '。\n')
                pair = []

    pair = []
    with open(r'data/dataset/total_dev.txt', 'w', encoding='utf-8') as f:
        for lyric in lyric_list[train_pair_len:train_pair_len + dev_pair_len]:
            pair.append(lyric)
            if len(pair) == 2:
                f.write(pair[0] + '， ' + pair[1] + '。\n')
                pair = []

    pair = []
    with open(r'data/dataset/total_test.txt', 'w', encoding='utf-8') as f:
        for lyric in lyric_list[train_pair_len + dev_pair_len:]:
            pair.append(lyric)
            if len(pair) == 2:
                f.write(pair[0] + '， ' + pair[1] + '。\n')
                pair = []


def data_process(file_dir, data_type):
    file = os.path.join(file_dir, "%s.txt" % data_type)
    input_file = os.path.join(file_dir, "%s.src" % data_type)
    output_file = os.path.join(file_dir, "%s.tgt" % data_type)
    skip_cnt = 0
    with open(file, 'r', encoding='utf-8') as f, open(input_file, 'w', encoding='utf-8') as inputf, open(output_file, "w", encoding='utf-8') as outputf:
        for input_output_pair in f:
            input_output_pair = input_output_pair.split('，')
            if len(input_output_pair) == 1:
                skip_cnt += 1
                continue
            inputf.write(input_output_pair[0].strip() + '\n')
            # zh_words = list(jieba.cut(en_zh_pair[1]))
            # zf.write(' '.join(zh_words) + '\n')
            outputf.write(input_output_pair[1].replace('。', '').strip() + '\n')
    print('skip_cnt in %s is %d' % (data_type, skip_cnt))


if __name__ == '__main__':
    data_preprocess()
    for data_type in ['total_train', 'total_dev', 'total_test']:
        file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
        data_process(file_dir, data_type)
