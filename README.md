perl
Copy code
# Whisper: a Chinese Speech Recognition System based on Transformers

## Installation
- Create and activate a new conda environment:
    ```
    conda create -n whisper python=3.8
    conda activate whisper
    ```

- Install required packages:
    ```
    pip install -r requirements.txt
    pip install git+https://github.com/huggingface/transformers
    ```

## Data Preparation
- Process the data using the same method as Wenet, generate `data.list` for training and validation sets:
    ```
    python tools/make_raw_list.py $data/wav.scp $data/text $data/data.list
    ```

## Training
- Start the training process:
    ```
    sbatch -o log ./run.sh
    ```

## Decoding
- Perform decoding:
    ```
    ./run.sh
    ```