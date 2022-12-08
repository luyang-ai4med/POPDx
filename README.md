# POPDx: An Automated Framework for Patient Phenotyping across 392,246 Individuals in the UK Biobank Study 
POPDx (Population-based Objective Phenotyping by Deep Extrapolation) is a bilinear machine learning framework for simultaneous multi-phenotype recognition. For additional information, please refer to our preprint, available at https://arxiv.org/abs/2208.11223. 

<img src="blob/overview.jpg" width="600" >

## Tools for UK Biobank

## Installation
Please clone our github repository as follows:
```
git clone https://github.com/luyang-ai4med/POPDx.git
```
## Dependencies
POPDx is developed in Python 3. We provide an conda environment containing the necessary dependencies. 
For our tests, we suggest using a single GPU (e.g. NVIDIA Tesla V100 SXM2 16 GB). 
```
conda env create -f popdx.yml
conda activate popdx
```
## POPDx training
POPDx can be explored and run through the command lines as follows: 
```
python code/POPDx_train.py -h
python code/POPDx_train.py -d './save/POPDx_train' 
```
Additional parameters can be defined by the user. 

```
The script to train POPDx. 
Please specify the train/val datasets path in the python script.

optional arguments:
  -h, --help            show this help message and exit
  -d SAVE_DIR, --save_dir SAVE_DIR
                        The folder to save the trained POPDx model e.g.
                        "./save/POPDx_train"
  -s HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        Default hidden size is 150.
  --use_gpu USE_GPU     Default setup is to use GPU.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Default learning rate is 0.0001
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        Default learning rate is 0
```
## POPDx testing
