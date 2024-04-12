# Train GPT with Colossal-AI

This example shows how to use [Colossal-AI](https://github.com/hpcaitech/ColossalAI) to run huggingface GPT training in distributed manners.

## GPT

We use the [GPT-2](https://huggingface.co/gpt2) model from huggingface transformers. The key learning goal of GPT-2 is to use unsupervised pre-training models to do supervised tasks.GPT-2 has an amazing performance in text generation, and the generated text exceeds people's expectations in terms of contextual coherence and emotional expression.

## Requirements

Before you can launch training, you need to install the following requirements.

### Install PyTorch

```bash
#conda
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
#pip
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### [Install Colossal-AI](https://github.com/hpcaitech/ColossalAI#installation)


### Install requirements

```bash
pip install -r requirements.txt
```

This is just an example that we download PyTorch=1.12.0, CUDA=11.6 and colossalai. You can download another version of PyTorch and its corresponding ColossalAI version. Just make sure that the version of ColossalAI is at least 0.1.10, PyTorch is at least 1.8.1 and transformers is at least 4.231.
If you want to test ZeRO1 and ZeRO2 in Colossal-AI, you need to ensure Colossal-AI>=0.1.12.

## Dataset

For simplicity, the input data is randomly generated here.

## Training
We provide two stable solutions.
One utilizes the Gemini to implement hybrid parallel strategies of Gemini, DDP/ZeRO, and Tensor Parallelism for a huggingface GPT model.
The other one use [Titans](https://github.com/hpcaitech/Titans), a distributed executed model zoo maintained by ColossalAI,to implement the hybrid parallel strategies of TP + ZeRO + PP.

We recommend using Gemini to quickly run your model in a distributed manner.
It doesn't require significant changes to the model structures, therefore you can apply it on a new model easily.
And use Titans as an advanced weapon to pursue a more extreme performance.
Titans has included the some typical models, such as Vit and GPT.
However, it requires some efforts to start if facing a new model structure.

### GeminiDPP/ZeRO + Tensor Parallelism
```bash
bash run_gemini.sh
```

### Hybridparallelism

Quick run
```bash
cd ./hybridparallelism
bash run.sh
```

### Troubleshooting
I tried to run my code in WSL


I've installed the cuda 


nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0



Below is my GPU


nvidia-smi
Fri Apr 12 21:44:44 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 551.78         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4070        On  |   00000000:01:00.0  On |                  N/A |
|  0%   34C    P8             10W /  215W |    1072MiB /  12282MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A        33      G   /Xwayland                                   N/A      |
+-----------------------------------------------------------------------------------------+


But it still give me below error


RuntimeError: No CUDA GPUs are available

