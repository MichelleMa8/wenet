# WeNet & ColossalAI

## Step 1: Requirements

``` sh
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
conda install pytorch=1.10.0 torchvision torchaudio=0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

## Step 2: Install WeNet (Python Only)

If you just want to use WeNet as a python package for speech recognition application,
just install it by `pip`, please note python 3.6+ is required.
``` sh
pip3 install wenetruntime
```

And please see [doc](runtime/binding/python/README.md) for usage.

## Step 3: Install ColossalAI

``` sh
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install colossalai
CUDA_EXT=1 pip install .
```

## Training

The script `./run.sh` is provided to train the model with or without ColossalAI's ZeRO optimizer.

- You can set `CUDA_VISIBLE_DEVICES` (line 6) to your available cuda devices
- You can change `data_dir` (line 13) to your own data directory
- You can change `gemini_state=true` or `gemini_state=false` (line 9) to decide whether using ColossalAI
- You can change `train_config` (line 17) to the model's configuration file.

