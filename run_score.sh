export CUDA_VISIBLE_DEVICES=$1
export ref=$2
export hypo=$3
export src=$4
export output=$5

rm $output/output.txt

python score.py \
--ref $ref \
--hypo $hypo \
--src $src \
--bart_score_cnn \
--bert_score \
--output $output

bash /data/home/xiekeli/factcc/run.sh $1 $hypo $src $output