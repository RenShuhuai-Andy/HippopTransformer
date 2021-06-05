TEXT=data/src_tgt

fairseq-preprocess \
  --source-lang src --target-lang tgt \
  --joined-dictionary \
  --trainpref $TEXT/total_train --validpref $TEXT/total_dev \
  --testpref $TEXT/total_test \
  --nwordssrc 24576 --nwordstgt 24576 \
  --destdir data/data-bin \
  --workers 20
