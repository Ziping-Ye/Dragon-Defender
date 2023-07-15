"""
    This script does all tasks to prepare the pretrain, train and validation dataset. 
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import mapping, exclude_attacks


def prepare_one_dataset(num_exclude, 
                        version, 
                        attack_names,
                        attack_names_to_count,
                        feature_list, 
                        traces_directory,
                        val_portion, 
                        sliding_window_length,
                        base_dir=r"/home/zipingye/cellular-ids"):

    print("******************************************************")
    print(f"exclude {num_exclude} attacks, version {version}")
    
    out_dir_name = f"exclude_{num_exclude}_attacks_version_{version}"

    out_dir = os.path.join(base_dir, "traces", out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # create dataframe to store different results
    pretrain_df = pd.DataFrame()
    train_df = pd.DataFrame()
    validation_df = pd.DataFrame()
    
    benign_traces = [os.path.join(traces_directory["benign"], benign_trace) for benign_trace in os.listdir(traces_directory["benign"])]
    attack_traces = [os.path.join(traces_directory["attack"], attack_trace) for attack_trace in os.listdir(traces_directory["attack"])]

    # want to construct a validation traces where each attack type is included
    good_val_attack_trace = False
    
    while not good_val_attack_trace: 
        print("trying to prepare a validation dataset where each type of attck is included")
        # randomly split benign traces into pool A and pool B
        val_benign_traces = random.sample(benign_traces, int(len(benign_traces) * val_portion)) # pool B
        val_benign_trace_names = [os.path.basename(val_benign_trace)[:-4] for val_benign_trace in val_benign_traces]
        # print("val_benign_traces length: ", len(val_benign_traces))
        # split attack traces accordingly
        val_attack_traces = [] # pool B
        for val_benign_trace_name in val_benign_trace_names:
            for attack_trace in attack_traces:
                if val_benign_trace_name in os.path.basename(attack_trace):
                    val_attack_traces.append(attack_trace)

        print("val_attack_traces number: ", len(val_attack_traces))
            
        # keep track of the trace number of each attack
        val_attack_trace_count = dict.fromkeys(attack_names, 0)

        for val_attack_trace in val_attack_traces:
            trace_attack = os.path.basename(val_attack_trace).split("+")[1]
            if "IMSI_Cracking_Reduced" in trace_attack: # deal with "IMSI_Cracking" and "IMSI_Cracking_Reduced" separately
                val_attack_trace_count["IMSI_Cracking_Reduced"] += 1
            elif "IMSI_Cracking" in trace_attack:
                val_attack_trace_count["IMSI_Cracking"] += 1
            else:
                for attack_name in attack_names_to_count:
                    if attack_name in trace_attack:
                        val_attack_trace_count[attack_name] += 1
                        break # move on to next trace    
        
        enough_attack_trace_count = 0
        # check if each attack is included
        for trace_count in val_attack_trace_count.values():
            if trace_count > 50:
                enough_attack_trace_count += 1
        
        if enough_attack_trace_count == len(attack_names):
            good_val_attack_trace = True
        else:
            print(val_attack_trace_count)

    # the complement of candidate validation traces will be used for pretrain and train
    pretrain_train_traces = [b_trace for b_trace in benign_traces if b_trace not in val_benign_traces] + \
                            [a_trace for a_trace in attack_traces if a_trace not in val_attack_traces] # pool A
    val_traces = val_attack_traces + val_benign_traces
    
    print("number of pretrain_train_traces: ", len(pretrain_train_traces))
    print("number of validation traces: ", len(val_traces))
    print("statistics of each attack type: (validation)", val_attack_trace_count)

    # *** construct a validation set ***
    print("\n\n***** construct a validation set *****")
    for val_trace in val_traces:
        try:
            filename = os.path.basename(val_trace)
            converted_trace_path = os.path.join(base_dir, "traces/train", filename)
            df = pd.read_csv(converted_trace_path)
            validation_df = pd.concat([validation_df, df])

        except FileNotFoundError: # some traces got filtered out during sliding window/trace2example
            continue

        except Exception as e:
            error_msg = f"{e} when reading {val_trace} during construct a candidate validation set"
            print(error_msg)
            continue
    
    # validation
    validation_df.to_csv(os.path.join(out_dir, "validation.csv"), index=False)

    # *** before preparing windows for pretrain, exclude certain number of attacks from pretrain and train dataset ***
    filtered_pretrain_train_traces, dropped_pretrain_train_trace_list, excluded_attacks = exclude_attacks(pretrain_train_traces, num_exclude, attack_names)

    # given the huge size of filtered_pretrain_train_traces, only need a portion for pretrain
    pretrain_portion = 0.15
    filtered_pretrain_train_traces = random.sample(filtered_pretrain_train_traces, k=int(len(filtered_pretrain_train_traces)*pretrain_portion))

    # *** prepare windows for pretrain ***
    print("\n\n***** prepare windows for pretrain *****")
    for pretrain_trace in tqdm(filtered_pretrain_train_traces):
        try:
            filename = os.path.basename(pretrain_trace)
            slided_trace_path = os.path.join(base_dir, "traces/pretrain", filename)
            df = pd.read_csv(slided_trace_path)
            pretrain_df = pd.concat([pretrain_df, df])
        
        except FileNotFoundError: # some traces got filtered out during sliding window/trace2example
            continue

        except Exception as e:
            error_msg = f"{e} when reading {pretrain_trace} during prepare windows for pretrain"
            print(error_msg)
            continue

    print("before removing repeating windows: ", len(pretrain_df))
    pretrain_len_before_removing_repeating_windows = len(pretrain_df)
    # remove repeating windows (all features and label are the same)
    pretrain_df.drop_duplicates(subset=["all_features", "label"], inplace=True)
    print("after removing repeating windows: ", len(pretrain_df))
    pretrain_len_after_removing_repeating_windows = len(pretrain_df)

    print("before removing conflicting windows: ", len(pretrain_df))
    pretrain_len_before_removing_conflicting_windows = len(pretrain_df)
    # treat conflicting windows (all features are the same, but label are different) as benign windows
    pretrain_df.loc[pretrain_df.duplicated(subset=["all_features"], keep=False), "label"] = 0 # mark all conflicting window as benign window, effectively turns conflicting window into repeating window

    conflicting_windows = pretrain_df[pretrain_df.duplicated(keep='first')] # Mark duplicates as True except for the first occurrence
    with pd.option_context('mode.chained_assignment', None):
        conflicting_windows.drop_duplicates(inplace=True) # remove duplicates within all the conflicting windows
    # write conflicting windows for human investigation
    conflicting_windows.to_csv(os.path.join(out_dir, "conflicting_windows.csv"), index=False)

    pretrain_df.drop_duplicates(subset=["all_features"], inplace=True) # remove conflicting windows
    print("after removing conflicting windows: ", len(pretrain_df))
    pretrain_len_after_removing_conflicting_windows = len(pretrain_df)
    # log relevant statistics

    total_benign_windows = pretrain_df.loc[:, "label"].tolist().count(0)
    total_attack_windows = pretrain_df.loc[:, "label"].tolist().count(1)
    print("statistics of the total pretrain windows: ", total_benign_windows, " benign widnows; ", total_attack_windows, " attack windows")

    # only need a portion
    pretrain_benign_windows = pretrain_df[pretrain_df["label"] == 0].sample(frac=0.1)
    pretrain_attack_windows = pretrain_df[pretrain_df["label"] == 1]
    pretrain = pd.concat([pretrain_benign_windows, pretrain_attack_windows])
    
    benign_windows = pretrain.loc[:, "label"].tolist().count(0)
    attack_windows = pretrain.loc[:, "label"].tolist().count(1)
    print("we only need a portion: ", benign_windows, " benign windows; ", attack_windows, " attack windows")
    # supervised contrastive pre-training
    pretrain.to_csv(os.path.join(out_dir, "pretrain.csv"), index=False)

    # visualization dataframe
    # unseen attacks
    if num_exclude > 0:
        unseen_df = pd.DataFrame()
        for unseen_trace in tqdm(dropped_pretrain_train_trace_list):
            try:
                filename = os.path.basename(unseen_trace)
                slided_trace_path = os.path.join(base_dir, "traces/pretrain", filename)
                df = pd.read_csv(slided_trace_path)
                unseen_df = pd.concat([unseen_df, df])
            
            except FileNotFoundError: # some traces got filtered out during sliding window/trace2example
                continue

            except Exception as e:
                error_msg = f"{e} when reading {unseen_trace} during prepare unseen attacks for visualization"
                print(error_msg)
                continue

        vis_df = pd.concat([pretrain.sample(frac=0.05), unseen_df.sample(frac=0.005)])
    
    else:
        vis_df = pretrain.sample(frac=0.05)

    vis_df.drop_duplicates(subset=["all_features", "label"], inplace=True)
    print("we will use ", len(vis_df), " windows for visualization")
    if num_exclude > 0:
        print("including ", len(unseen_df), " windows from dropped traces")
    
    vis_df.to_csv(os.path.join(out_dir, "visualization.csv"), index=False)

    # *** prepare traces for train ***
    print("\n\n***** prepare traces for train *****")
    for train_trace in tqdm(filtered_pretrain_train_traces):
        try:
            filename = os.path.basename(train_trace)
            converted_trace_path = os.path.join(base_dir, "traces/train", filename)
            df = pd.read_csv(converted_trace_path)
            train_df = pd.concat([train_df, df])
    
        except FileNotFoundError: # some traces got filtered out during sliding window/trace2example
            continue
   
        except Exception as e:
            error_msg = f"{e} when reading {train_trace} during prepare traces for train"
            print(error_msg)
            continue
    
    print("there are ", len(train_df), " traces for training in total")
    train = train_df.sample(frac=0.2)
    print("we keep ", len(train), " as this should be enough to train a light weight model")
    # train
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)

    # log relevant statistics
    with open(os.path.join(out_dir, "prepare_data_log.txt"), "w") as f:
        f.write("number of pretrain_train_traces: " + str(len(pretrain_train_traces)) + "\n")
        f.write("number of validation traces: " + str(len(val_traces)) + "\n")
        f.write("statistics of each attack type: (validation)\n")
        for k, v in val_attack_trace_count.items():
            f.write(k + " " + str(v) + "\n")
        f.write("while preparing windows for pretrain\n")
        f.write("there are " + str(pretrain_len_before_removing_repeating_windows) + " windows before removing repeating windows\n")
        f.write("there are " + str(pretrain_len_after_removing_repeating_windows) + " windows after removing repeating windows\n")
        f.write("there are " + str(pretrain_len_before_removing_conflicting_windows) + " windows before removing conflicting windows\n")
        f.write("there are " + str(pretrain_len_after_removing_conflicting_windows) + " windows after removing conflicting windows\n")
        f.write("in total, " + str(total_benign_windows) + " benign windows and " + str(total_attack_windows) + " attack windows\n")
        f.write("we will use " + str(len(pretrain)) + " windows for pretraining\n")
        f.write("there are " + str(benign_windows) + " benign windows and " + str(attack_windows) + " attack windows\n")
        f.write("we will use " + str(len(vis_df)) + " windows for visualization\n")
        if num_exclude > 0:
            f.write("including " + str(len(unseen_df)) + " windows from dropped traces\n")
        f.write("in total, there are " + str(len(train_df)) + " traces that could be used for training\n")
        f.write("we will use " + str(len(train)) + " traces to train a light weight model\n")
        f.write("these attacks were excluded from pretraining and training\n")
        f.write("\t".join(excluded_attacks))

    print(f"DONE --> exclude {num_exclude} attacks, version {version} <-- DONE")
    print("********************************************************")

    return out_dir_name


if __name__ == "__main__":

    val_portion = 0.1
    attack_names = ["AKA_Bypass", "Attach_Reject", "EMM_Information", "IMEI_Catching",
                    "IMSI_Catching", "Malformed_Identity_Request", "Null_Encryption",
                    "Numb_Attack", "Repeat_Security_Mode_Command", "RLF_Report",
                    "Service_Reject", "TAU_Reject"]

    feature_list = ["message name", "Attach with IMSI", "Null encryption", 
                    "Enable IMEISV", "Cell ID", "TAC", "EMM state", 
                    "EMM substate", "EMM cause", "paging_record_number"]

