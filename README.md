# Dragon Defender: Runtime Threat Detection for Cellular Devices

This repository contains the implementation for Ziping Ye's undergraduate honors thesis, "Dragon Defender: Runtime Threat Detection for Cellular Devices."

## Useful Links
- [Dragon Defender: Runtime Threat Detection for Cellular Devices](https://www.dropbox.com/s/zllft80tsinsmb9/Dragon%20Defender-%20Runtime%20Threat%20Detection%20for%20Cellular%20Devices.pdf?dl=0)
- [Raw Traces](https://www.dropbox.com/sh/64ja37yhip60jre/AAByBoOVjqk5TY-bN7yMZyzSa?dl=0)
- [Processed Traces](https://www.dropbox.com/sh/y26vhn557ts3hpj/AAAsYWriLB7jWBbqEcjOAH7sa?dl=0)
- [Trained Model](https://www.dropbox.com/sh/3x2w56agrvicue4/AACkJSF4HbiyUV-C2WxZ4aFHa?dl=0)

## Instructions
Please follow the instructions below to use and reproduce Dragon Defender's results.

### Step 1: Process Traces
Let's start from raw traces - each example is a message (containing all feature values) and a label. 
```
cd 
```

To train the Window Encoder, each train/test example is a sliding window (i.e., 31 consecutive messages, right padded), and a window label. Execute 
```
python sliding_window.py
```
to construct sliding windows. The results will be saved in the directory `traces/pretrain`.

To train the Message Tagger, each train/test example is the sequence of messages in one session. Execute
```
python trace2example.py
```
to construct train/test examples. The results will be saved in the directory `traces/train`.


### Step 2: Prepare Dataset
Now, we can prepare the training and test set. Execute
```
python prepare_dataset.py
```
which takes care of constructing datasets for training, testing, and visualization. A new folder `exclude_{num_exclude}_attacks_version_{version}` will be created under the parent folder `traces`. In this folder, you will find five csv files
- validation.csv
- conflicting_windows.csv
- pretrain.csv
- visualization.csv
- train.csv


### Step 3: Model Training
Switch back to the project directory
```
cd ..
```

You can train the Window Encoder (Projection BERT) model by executing
```
python pretrain.py
```
and train the Message Tagger (LSTM model) model by executing
```
python train.py
```
A directory called `logs` will be automatically created by PyTorch Lightning and trained models will be saved there.


### Step 4: Evaluation and Visualization
You can visualize the embedding space learned by Window Encoder in 2-dim and 3-dim space by executing
```
python visualization.py
```

### Shortcut
Steps 2 - 4 are implemented in `main.py`. If you want to automate the train and evaluation process, you can execute
```
python main.py
```
instead of going through steps 2-4.

