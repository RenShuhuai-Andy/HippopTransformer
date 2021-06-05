import os
import json
import jieba


def data_process(file_dir, data_type):
    file = os.path.join(file_dir, "%s.txt" % data_type)
    input_file = os.path.join(file_dir, "%s.src" % data_type)
    output_file = os.path.join(file_dir, "%s.tgt" % data_type)
    with open(file, 'r', encoding='utf-8') as f, open(input_file, 'w', encoding='utf-8') as inputf, open(output_file, "w", encoding='utf-8') as outputf:
        for input_output_pair in f:
            input_output_pair = input_output_pair.split('，')
            inputf.write(input_output_pair[0].strip() + '\n')
            # zh_words = list(jieba.cut(en_zh_pair[1]))
            # zf.write(' '.join(zh_words) + '\n')
            outputf.write(input_output_pair[1].replace(' 。', '').strip() + '\n')


if __name__ == '__main__':
    for data_type in ['total_train', 'total_dev', 'total_test']:
        file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
        data_process(file_dir, data_type)
