# Dragon Defender: Runtime Threat Detection for Cellular Devices

This repository contains the implementation and evaluation for the paper "Dragon Defender: Runtime Threat Detection for Cellular Devices."


## Instructions
Please follow the instructions below to use and reproduce Dragon Defender's results.

### Step 1: Process Traces
Let's start from raw traces - each example is a message (containing all feature values) and a label. 

- Download the traces from the `raw traces` link above
- Unzip the downloaded file
- Create a folder `traces` and move the unzipped traces there

```
cd trace_process
```

To train the Window Encoder, each train/test example is a sliding window (i.e., 31 consecutive messages, right padded) and a window label. Execute 
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
which takes care of constructing datasets for training, testing, and visualization. A new folder `exclude_{num_exclude}_attacks_version_{version}` will be created under the parent folder `traces`. In this folder, you will find five CSV files
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
A directory called `logs` will be automatically created by PyTorch Lightning, and trained models will be saved there.


### Step 4: Evaluation and Visualization
You can visualize the embedding space learned by Window Encoder in 2-dim and 3-dim space by executing
```
python visualization.py
```

### Shortcut
Steps 2 - 4 are implemented in `main.py`. If you want to automate the training and evaluation process, you can execute
```
python main.py
```
instead of going through steps 2-4.

