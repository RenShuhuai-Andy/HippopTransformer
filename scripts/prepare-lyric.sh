#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000


if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

OUTDIR=data/src_tgt
src=src
tgt=tgt
prep=$OUTDIR
tmp=$prep/tmp
jieba=$tmp/jieba
orig=data/dataset

mkdir -p $orig $tmp $prep $jieba

echo "pre-processing data..."
python data/data_process.py


echo "tokenizing zh data by jieba..."
for l in $src $tgt; do
    for d in total_train total_dev total_test; do
        cp $orig/$d.$l $tmp
        python -m jieba -d " " $tmp/$d.$l > $jieba/$d.$l
    done
done

echo "tokenizing data..."
for l in $src $tgt; do
  for d in total_train total_dev total_test; do
      cat $jieba/$d.$l | \
          perl $TOKENIZER -threads 8 -a -l zh > $tmp/$d.$l
  done
done

TRAIN=$tmp/total_train.src-tgt
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/total_train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in total_train.$L total_dev.$L total_test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.total_train $src $tgt $prep/total_train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.total_dev $src $tgt $prep/total_dev 1 250

for L in $src $tgt; do
    cp $tmp/bpe.total_test.$L $prep/total_test.$L
done

echo "reverse text..."
python data/reverse_text.py