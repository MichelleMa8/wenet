#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="6,7"

# Whether to use Gemini Optimizer, 'true' or 'false'
gemini_state=false
# wav data dir
wave_data=data
data_url=www.openslr.org/resources/12
data_dir=/data/scratch/librspeech/

# Optional train_config
# 1. conf/train_transformer_large.yaml: Standard transformer
train_config=examples/librispeech/s0/conf/train_u2++_conformer.yaml
checkpoint=
cmvn=true
do_delta=false
dir=exp/sp_spec_aug
# use average_checkpoint will get better resulst
average_checkpoint=true
decode_checkpoint=$dir/final.pt
# maybe you can try to adjust it if you can not get close results as README.md
average_num=10
decode_modes="attention_rescoring ctc_greedy_search ctc_prefix_beam_search attention"
# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram
set -e
set -u
set -o pipefail
train_set=dev
dev_set=dev
recog_set="test_clean test_other dev_clean dev_other"

# Download data (training data too large; use dev temporarily)
# for part in train-clean-100; do
# ./examples/librispeech/s0/local/download_and_untar.sh ${data_dir} ${data_url} ${part}
# done

# # Prepare training data (use dev instead for memory space's sake)
# echo "stage 0: Data preparation"
# for part in dev-clean test-clean dev-other test-other; do
# # # use underscore-separated names in data directories.
# ./examples/librispeech/s0/local/data_prep_torchaudio.sh ${data_dir}/LibriSpeech/${part} $wave_data/${part//-/_}
# done


# ### Task dependent. You have to design training and dev sets by yourself.
# ### But you can utilize Kaldi recipes in most cases
# echo "stage 1: Feature Generation"
# mkdir -p $wave_data/train_960
# # merge total training data
# for set in dev-clean; do
# for f in `ls $wave_data/$set`; do
#     cat $wave_data/$set/$f >> $wave_data/train_960/$f
# done
# done
# mkdir -p $wave_data/dev
# # merge total dev data
# for set in dev_clean dev_other; do
# for f in `ls $wave_data/$set`; do
#     cat $wave_data/$set/$f >> $wave_data/dev/$f
# done
# done

# ./examples/librispeech/s0/tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
# --in_scp $wave_data/$train_set/wav.scp \
# --out_cmvn $wave_data/$train_set/global_cmvn

dict=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
# ### Task dependent. You have to check non-linguistic symbols used in the corpus.
# echo "stage 2: Dictionary and Json Data Preparation"
# mkdir -p data/lang_char/

# echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
# echo "<unk> 1" >> ${dict} # <unk> must be 1

# # we borrowed these code and scripts which are related bpe from ESPnet.
# cut -f 2- -d" " $wave_data/${train_set}/text > $wave_data/lang_char/input.txt
# ./examples/librispeech/s0/tools/spm_train --input=$wave_data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
# ./examples/librispeech/s0/tools/spm_encode --model=${bpemodel}.model --output_format=piece < $wave_data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
num_token=$(cat $dict | wc -l)
echo "<sos/eos> $num_token" >> $dict # <eos>
wc -l ${dict}

# # Prepare wenet required data
# echo "Prepare data, prepare required format"
# for x in dev ${recog_set} $train_set ; do
# ./examples/librispeech/s0/tools/make_raw_list.py $wave_data/$x/wav.scp $wave_data/$x/text \
#     $wave_data/$x/data.list
# done

# Training script
mkdir -p $dir
INIT_FILE=$dir/ddp_init
rm -f $INIT_FILE # delete old one before starting
init_method=file://$(readlink -f $INIT_FILE)
echo "$0: init method is $init_method"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
# Use "nccl" if it works, otherwise use "gloo"
dist_backend="gloo"
cmvn_opts=
$cmvn && cmvn_opts="--cmvn $wave_data/${train_set}/global_cmvn"

# train.py will write $train_config to $dir/train.yaml with model input
# and output dimension, train.yaml will be used for inference or model
# export later
if [[ $gemini_state == t* ]] || [[ $gemini_state == T* ]] || [ $gemini_state == 1 ]; then
    echo "USING COLOSSALAI"
    for ((i = 0; i < $num_gpus; ++i)); do
    {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    python wenet/bin/train.py --gpu $gpu_id \
        --config $train_config \
        --data_type raw \
        --symbol_table $dict \
        --train_data $wave_data/$train_set/data.list \
        --cv_data $wave_data/dev/data.list \
        ${checkpoint:+--checkpoint $checkpoint} \
        --model_dir $dir \
        --ddp.init_method $init_method \
        --ddp.world_size $num_gpus \
        --ddp.rank $i \
        --ddp.dist_backend $dist_backend \
        --num_workers 1 \
        $cmvn_opts \
        --pin_memory \
        --rank $i \
        --world_size $num_gpus \
        --port 28600 \
        --host localhost \
        --gemini $gemini_state
    } &
    done
    wait
else
    echo "NOT USING COLOSSALAI"
    for ((i = 0; i < $num_gpus; ++i)); do
    {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    python wenet/bin/train.py --gpu $gpu_id \
        --config $train_config \
        --data_type raw \
        --symbol_table $dict \
        --train_data $wave_data/$train_set/data.list \
        --cv_data $wave_data/dev/data.list \
        ${checkpoint:+--checkpoint $checkpoint} \
        --model_dir $dir \
        --ddp.init_method $init_method \
        --ddp.world_size $num_gpus \
        --ddp.rank $i \
        --ddp.dist_backend $dist_backend \
        --num_workers 1 \
        $cmvn_opts \
        --pin_memory \
        --gemini $gemini_state \
        --rank $i \
        --world_size $num_gpus
    } &
    done
    wait
fi