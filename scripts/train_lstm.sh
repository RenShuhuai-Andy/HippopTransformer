model_signature=lstm_wiseman_iwslt_de_en
GPU="4,5,6,7"
save_tag=${model_signature}

CUDA_VISIBLE_DEVICES=$GPU fairseq-train \
    data/data-bin \
    --arch $model_signature  \
    --dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 \
    --lr 0.005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 4096 \
    --save-dir checkpoints/$save_tag \
    --max-update 25000 --save-interval-updates 2000  --validate-interval 3 \
    --keep-interval-updates 10 \
    --keep-best-checkpoints 10 \
    --no-epoch-checkpoints \
    --tensorboard-logdir tensorboard-logdir/$save_tag \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 2>&1 | tee -a logs/$save_tag.log