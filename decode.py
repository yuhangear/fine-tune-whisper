from datasets import load_dataset, DatasetDict


import torch
from torch.utils.data import DataLoader
import os

from wenet.dataset.dataset import Dataset
from wenet.dataset.dataset import Dataset_dev
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def avg_parameters2():
    
    models = []
    with open("/home/yuhang001/eng_whisper/whisper-small-eng3/need_path") as f :
        for i in f:
            i=i.strip()
            models.append(    torch.load(i +"/pytorch_model.bin")  )


    avg_model = torch.load("/home/yuhang001/eng_whisper/whisper-small-eng3/checkpoint-avg/pytorch_model.bin")
    for key in avg_model:
        avg_model[key] = torch.true_divide(sum([_[key] for _ in models]), len(models))
    torch.save(avg_model,  "/home/yuhang001/eng_whisper/whisper-small-eng3/checkpoint-avg/pytorch_model.bin")


# avg_parameters2()





import sys
dev_data = sys.argv[1]
log_result = sys.argv[2]
model_dir=sys.argv[3]



dev_dataset = Dataset_dev(dev_data,"english")


from transformers import AutoProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

model = WhisperForConditionalGeneration.from_pretrained(model_dir).to("cuda")

forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")




#自定义dataset解码
ref=open(log_result+"_decoder_ntu_ref3","w")
reg=open(log_result+"_decoder_ntu_reg3","w")
index=1
for i in dev_dataset:
    ref_utt=processor.batch_decode(torch.tensor(i["labels"]).unsqueeze(0).to("cuda")   , skip_special_tokens=True)[0] 
    ref_utt=processor.tokenizer._normalize(ref_utt)

    ref.writelines("utt_" + str(index).zfill(5)  + " " +str(ref_utt)+"\n")
    input=i["input_features"]
   
    with torch.no_grad():
        generated_ids = model.generate(inputs=input.to("cuda") ,forced_decoder_ids=forced_decoder_ids ) #forced_decoder_ids=forced_decoder_ids
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    transcription=processor.tokenizer._normalize(transcription)
    reg.writelines("utt_" + str(index).zfill(5)  + " " +str(transcription)+"\n")
    print(index)
    index=index+1
ref.close()
reg.close()


# python tools/compute-wer.py --char 1 --v 1  /home/yuhang001/eng_whisper/decoder_ntu_ref   /home/yuhang001/eng_whisper/decoder_ntu_reg 




