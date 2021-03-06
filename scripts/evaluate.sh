while getopts ":m:c:n:" opt
do
    case $opt in
        m)
          echo "model_signature=$OPTARG"
          model_signature=$OPTARG
        ;;
        c)
          echo "CUDA_VISIBLE_DEVICES=$OPTARG"
          CUDA_VISIBLE_DEVICES=$OPTARG
        ;;
        n)
          echo "num_epoch_checkpoints=$OPTARG"
          num_epoch_checkpoints=$OPTARG
        ;;
        ?)
          echo "unknown parameters"
        exit 1;;
    esac
done

OUTPUT_PATH=checkpoints/$model_signature
python hippop_transformer/utils/average_checkpoints.py --inputs $OUTPUT_PATH \
  --num-epoch-checkpoints $num_epoch_checkpoints --output $OUTPUT_PATH/avg_$num_epoch_checkpoints.pt

result_dir=results/$model_signature

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES fairseq-generate \
      data/data-bin \
      --user-dir hippop_transformer \
      --task translation \
      --results-path $result_dir/avg_$num_epoch_checkpoints \
      --path $OUTPUT_PATH/avg_$num_epoch_checkpoints.pt \
      --batch-size 256 \
      --remove-bpe \
      --beam 5

echo "evaluation on average_checkpoints $num_epoch_checkpoints:" | tee -a logs/$model_signature.log
tail -1 $result_dir/avg_$num_epoch_checkpoints/generate-test.txt | tee -a logs/$model_signature.log

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES fairseq-generate \
      data/data-bin \
      --user-dir hippop_transformer \
      --task translation \
      --results-path $result_dir \
      --path checkpoints/$model_signature/checkpoint_best.pt \
      --batch-size 256 \
      --remove-bpe \
      --beam 5

echo "evaluation on best checkpoint:" | tee -a logs/$model_signature.log
tail -1 $result_dir/generate-test.txt | tee -a logs/$model_signature.log

echo "rhyme rate on average_checkpoints $num_epoch_checkpoints:"
python hippop_transformer/utils/extract_generate_output_with_rhyme.py \
  --output $result_dir/avg_$num_epoch_checkpoints/generate-test \
  --srclang src --tgtlang tgt \
  $result_dir/avg_$num_epoch_checkpoints/generate-test.txt

echo "rhyme rate on best checkpoint:"
python hippop_transformer/utils/extract_generate_output_with_rhyme.py \
  --output $result_dir/generate-test \
  --srclang src --tgtlang tgt \
  $result_dir/generate-test.txt