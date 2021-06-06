model_signature=fconv
GPU="0,1,2,3"
save_tag=${model_signature}

CUDA_VISIBLE_DEVICES=$GPU fairseq-train \
    data/data-bin \
    --arch fconv_wmt_en_de  \
    --dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 4096 \
    --save-dir checkpoints/$save_tag \
    --validate-interval 3 \
    --keep-interval-updates 10 \
    --keep-best-checkpoints 10 \
    --no-epoch-checkpoints \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 2>&1 | tee -a logs/$save_tag.log