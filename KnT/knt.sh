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

# Datasets, optional choices include flore200, iwslt, CCMatrix, ccmt, etc.
test_name=ccmt 

# Language Pair
all_lang_pairs=(en-zh en-fa fa-en en-lo lo-en en-ur ur-en en-uz uz-en)

BS=200
KS_BS=100
export CUDA_VISIBLE_DEVICES=0

# Model path
MODEL_CKPT=/...../llm_models/$model_name

# Step 1: Construct prompts for knowledge generation 
# for lp in ${all_lang_pairs[@]}
# do
# 	echo $lp
#    src=${lp%%-*}
#    tgt=${lp##*-}

#    python3 $DATA/$model_name/format_ask_topic.py \
#        -w $WS \
#        -tn $test_name \
#        -m $model_name \
#        -s $src \
#        -t $tgt

#    python3 $DATA/$model_name/format_ask_demo.py \
#        -w $WS \
#        -tn $test_name \
#        -m $model_name \
#        -s $src \
#        -t $tgt

#    python3 $DATA/$model_name/format_ask_cod.py \
#        -w $WS \
#        -tn $test_name \
#        -m $model_name \
#        -s $src \
#        -t $tgt

# #    python3 $DATA/$model_name/format_ask_dw.py \
# #        -w $WS \
# #        -tn $test_name \
# #        -m $model_name \
# #        -s $src \
# #        -t $tgt

# done

# Step 2: Use the prompts constructed in Step 1 to generate the knowledge required for translating sentences in the source language 
#  for lp in ${all_lang_pairs[@]}
#  do
#     echo $model_name $lp
#     src=${lp%%-*}
#     tgt=${lp##*-}

#     for know in demo topic cod
#     do
#          python3 $WS/model/$model_name/translate.py \
#              --model-name-or-path $MODEL_CKPT \
#              -i $FORMAT/$model_name/$test_name.$lp/$test_name.$lp.$src.$know \
#              -o $OUTPUT/$model_name/$test_name.$lp/$test_name.$lp.$src.$know \
#              -sa beam \
#              --batch $BS \
#              --tp 1
#     done
#  done


# Step 3: Use the knowledge generated in Step 2 to construct prompts, so that the large language model can produce the corresponding translation based on the knowledge of the source language sentences.
# for lp in ${all_lang_pairs[@]}
# do
#   echo $lp
#   src=${lp%%-*}
#   tgt=${lp##*-}

#   for know in demo cod topic dw_LH
#   do
#       python3 $DATA/$model_name/format_$know.py \
#           -w $WS \
#           -tn $test_name \
#           -m $model_name \
#           -s $src \
#           -t $tgt
#   done
# done


# Step 4: Use the prompts created in Step 3 to make the large language model generate translations.
# for lp in ${all_lang_pairs[@]}
# do
#    echo $model_name $lp
#    src=${lp%%-*}
#    tgt=${lp##*-}

#     for know in demo cod topic dw
#     do
#         python3 $WS/model/$model_name/translate.py \
#             --model-name-or-path $MODEL_CKPT \
#             -i $FORMAT/with-knowledge/$model_name/$test_name.$lp/$test_name.$lp.$src.$know \
#             -o $OUTPUT/$model_name/$test_name.$lp/$test_name.$lp.$tgt.$know \
#             -sa beam \
#             --batch $BS \
#             --tp 1
#     done 
# done



# Step 5: Use QE evaluation metrics to filter multiple candidate translations
# for lp in ${all_lang_pairs[@]}
# do
#     src=${lp%%-*}
#     tgt=${lp##*-}


#     python3 $SCRIPT/knowledge-selection.py \
#         --sys   $OUTPUT/$model_name/$test_name.$lp/$test_name.$lp.$tgt.beam.0-shot \
#                 $OUTPUT/$model_name/$test_name.$lp/$test_name.$lp.$tgt.cod \
#                 $OUTPUT/$model_name/$test_name.$lp/$test_name.$lp.$tgt.topic \
#                 $OUTPUT/$model_name/$test_name.$lp/$test_name.$lp.$tgt.demo \
#                 $OUTPUT/$model_name/$test_name.$lp/$test_name.$lp.$tgt.dw \
#         --src   $RAW/$test_name.$lp.$src \
#         --ref   $RAW/$test_name.$lp.$tgt \
#         --out   $OUTPUT/$model_name/$test_name.$lp/$test_name.$lp.$tgt.knt_o_kiwi \
#         --src-lang $src --tgt-lang $tgt \
#         -bs     $KS_BS \
#         --metric comet_kiwi \
#         --comet-qe-model-name wmt23-cometkiwi-da-xl
# done

