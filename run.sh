#!/usr/bin/env bash

cmd="./slurm.pl --quiet --nodelist=node04"

train_data="/home/yuhang001/eng_whisper/train-nov-09_2/all_data.list_shuff_sub_eng_10time_sing_addagain_eng_shuff"
dev_data="/home/yuhang001/eng_whisper/train-nov-09_2/all_data.list_shuff_head_1000"
language = "english"
output_dir= "/home/yuhang001/eng_whisper/whisper-small-eng6_200"

$cmd --num-threads 12 --gpu 2  ./train.1.log_7_2000_id python fine-turn-whisper.py  $train_data  $dev_data $language $output_dir
