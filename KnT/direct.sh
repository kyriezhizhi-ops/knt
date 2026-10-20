set -e
set -u
BASEDIR=$(realpath `dirname $0`)
WS=$BASEDIR
DATA=$WS/data
FORMAT=$DATA/format
RAW=$DATA/raw
OUTPUT=$WS/output
SCRIPT=$WS/scripts

# Model name
model_name=Qwen3-32B

# Model path
MODEL_CKPT=..../llm_models/$model_name

# Datasets, optional choices include flore200, iwslt, CCMatrix, ccmt, etc.
test_name=ccmt 

# Language Pair
all_lang_pairs=(en-zh en-fa fa-en en-lo lo-en en-ur ur-en en-uz uz-en)

export CUDA_VISIBLE_DEVICES=0
BS=200
KS_BS=100

# Step 1: Construct the prompts required for 0-shot
for lp in ${all_lang_pairs[@]}
do
   src=${lp%%-*}
   tgt=${lp##*-}
   python3 $DATA/$model_name/format_base.py \
       -w $WS \
       -tn $test_name \
       -s $src \
       -t $tgt \
       -m $model_name \
       -n 0
done

# Step 2: Instruct the large language model to generate the corresponding translation based on the prompts created in Step 1
for lp in ${all_lang_pairs[@]}
do
   echo $model_name $lp
   src=${lp%%-*}
   tgt=${lp##*-}

    python3 $WS/model/$model_name/translate.py \
       --model-name-or-path $MODEL_CKPT \
       -i $FORMAT/$model_name/$test_name.$lp/$test_name.$lp.$src.0-shot \
       -o $OUTPUT/$model_name/$test_name.$lp/$test_name.$lp.$tgt.beam.0-shot \
       -b $BS \
       --search-algorithm beam \
       --tp 1
       
done