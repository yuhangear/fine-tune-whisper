项目名称
安装
环境要求
Python 3.8
安装依赖
sh
Copy code
conda create -n whisper python=3.8
conda activate whisper 
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers
准备数据
对数据进行处理时候是按照wenet的处理方式，训练和验证集按照wenet的方式进行处理，生成data.list
sh
Copy code
python tools/make_raw_list.py $data/wav.scp $data/text $data/data.list
训练模型
sh
Copy code
sbatch -o log ./run.sh
解码
sh
Copy code
./run.sh
参考文献
Wenet