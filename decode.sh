#!/usr/bin/env bash


dev_data1="/home/yuhang001/yuhang001/new_wenet/wenet/examples/gigaspeech/malay_eng/data/ntu-conversation/data.list"
dev_data2="/home/yuhang001/yuhang001/espnet/egs2/librispeech_100/asr1/data/test_clean/data.list"
dev_data3="/home/yuhang001/yuhang001/new_wenet/wenet/examples/gigaspeech/malay_eng/data/sherlock-en-noisy_05-mar2022/data.list"

#200小时，中英文结果 24 epoch
model="/home/yuhang001/eng_whisper/whisper-small-eng6_200/checkpoint-450"
#2000小时 中英文结果 8epoch
# model="/home/yuhang001/eng_whisper/whisper-small-eng6_2k/checkpoint-1650"

model=/home/yuhang001/eng_whisper/whisper-small-eng6_2000_id/checkpoint-1700

for dev_data in $dev_data1 $dev_data2 $dev_data3 ;do 

python decode.py $dev_data  ${dev_data}_log   $model
wait 
python tools/compute-wer.py --char 1 --v 1 ${dev_data}_log_decoder_ntu_reg3 ${dev_data}_log_decoder_ntu_ref3  > ${dev_data}_result_id

wait 

done  