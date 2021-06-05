import os
import json
import jieba


def data_process(file_dir, data_type, lang):
    texts_list = []
    file = os.path.join(file_dir, "%s.%s" % (data_type, lang))
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            line.reverse()
            texts_list.append(line)
    with open(file, 'w', encoding='utf-8') as f:
        for line in texts_list:
            f.write(' '.join(line) + '\n')


if __name__ == '__main__':
    for lang in ['src', 'tgt']:
        for data_type in ['total_train', 'total_dev', 'total_test']:
            file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src_tgt')
            data_process(file_dir, data_type, lang)
