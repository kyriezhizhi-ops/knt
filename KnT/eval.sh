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
model_names=Qwen3-32B

# Model path
MODEL=..../llm_models/$model_names

KS_BS=100
BATCH_SIZE=100

# Language Pair
all_lang_pairs=(en-zh en-fa fa-en en-lo lo-en en-ur ur-en en-uz uz-en)

# Datasets, optional choices include flore200, iwslt, CCMatrix, ccmt, etc.
test_name=ccmt 

export CUDA_VISIBLE_DEVICES=0

for model_name in ${model_names[@]}
do
    cd $OUTPUT/$model_name
    for lp in ${all_lang_pairs[@]}

    do
        src=${lp%%-*}
        tgt=${lp##*-}
        output_path=$WS/output/$model_names/$test_name.$src-$tgt
        madlad_path=$WS/output/$model_names/$test_name
        CAND0=$output_path/$test_name.$src-$tgt.$tgt.beam.0-shot
        CAND1=$output_path/$test_name.$src-$tgt.$tgt.demo
        CAND2=$output_path/$test_name.$src-$tgt.$tgt.cod
        CAND3=$output_path/$test_name.$src-$tgt.$tgt.topic
        CAND4=$output_path/$test_name.$src-$tgt.$tgt.dw
        CAND5=$output_path/$test_name.$src-$tgt.$tgt.knt_o_kiwi

        CANDS=($CAND5)
        src_file=$RAW/$test_name.$src-$tgt.$src
        ref_file=$RAW/$test_name.$src-$tgt.$tgt
        for CAND in ${CANDS[@]}
        do
            if [[ ! -s $CAND ]]
            then
                echo "ERROR: file not found $CAND"
                exit 1
            fi
        done

        # XComet-XL evaluation metrics
        echo "COMET: $lp $model_name"
        python3 $SCRIPT/score.py \
            --sys ${CANDS[@]} \
            --src $src_file \
            --ref $ref_file \
            --metric xcomet\
            --batch-size $BATCH_SIZE
        # BLEURT evaluation metric
        python3 $SCRIPT/score.py \
            --sys ${CANDS[@]} \
            --src $src_file \
            --ref $ref_file \
            --metric bleurt \
            --batch-size $BATCH_SIZE
    done
done