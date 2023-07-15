"""
    This script implements the sliding window on 4G LTE traces to get window of given width for pretraining,
    and also label each window according to the label of the last message in the window.
    
    Input: csv file where each row is a message
    Output: csv file(s) where each row is one pretrain example as well as its label
            - each cell is a feature sequence (= window_width)
"""

import json
import os
import sys
import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import mapping


def sliding_window(csv_path, out_path):

    df = pd.read_csv(csv_path)
    df_len = len(df)

    # store all windows from the given trace
    out = pd.DataFrame()

    for index, _ in df.iterrows():
        
        if index + sliding_window_length > df_len:
            break
        else:
            window = df.iloc[index : index+sliding_window_length, :]

        # window label = last message label 
        window_label = int(window.loc[:, "label"].tolist()[-1])

        with pd.option_context('mode.chained_assignment', None):
            # map tracking area code and cell id to 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h' within each window
            window.loc[:, "Cell ID"] = mapping(window.loc[:, "Cell ID"])
            window.loc[:, "TAC"] = mapping(window.loc[:, "TAC"])

        window_dict = dict.fromkeys(columns)
        all_features_str = ""

        for feature in feature_list:
            try:
                feature_str = " ".join([str(int(e)) if feature == "EMM cause" else str(e) for e in window.loc[:, feature].tolist()])
                window_dict[feature] = feature_str
                all_features_str += feature_str
            except KeyError as keyerr:
                sliding_window_error_list.append(f"{keyerr} occured in {csv_path} \n")
            except Exception as e:
                sliding_window_error_list.append(f"{e} occured in {csv_path} \n")

        # window label = last message label 
        window_dict["label"] = window_label
        window_dict["all_features"] = all_features_str

        out_new_row = pd.DataFrame(window_dict, index=[index])

        out = pd.concat([out, out_new_row])

    if len(out) > 0: # for very short trace, we won't get any sliding window
        # drop repeating windows within each trace
        out.drop_duplicates(subset=["all_features", "label"], inplace=True)
        out.to_csv(out_path, index=False)


if __name__ == "__main__":

    base_dir = os.getcwd()
    global_parameters = json.load(open(os.path.join(base_dir, "helpers/Global_Parameters.json")))

    feature_list = global_parameters['feature_list']
    sliding_window_length = global_parameters['sliding_window_length']
    # print("feature_list: ", feature_list)

    columns = feature_list + ["all_features", "label"]

    sliding_window_error_list = []
    sliding_window_error_file = os.path.join(base_dir, "log_files/sliding_window_error.txt")
    
    csv_dirs = [os.path.join(base_dir, "traces/benign_traces"), os.path.join(base_dir, "traces/attack_traces")]
    out_dir = os.path.join(base_dir, "traces/pretrain")
    os.makedirs(out_dir, exist_ok=True)

    for csv_dir in csv_dirs:

        csv_list = os.listdir(csv_dir)

        for csv_file in tqdm(csv_list):
            try:
                csv_path = os.path.join(csv_dir, csv_file)  
                out_path = os.path.join(out_dir, csv_file)
                sliding_window(csv_path, out_path)

            except Exception as e:
                error_msg = f"{e} occured in {csv_path} \n"
                print(error_msg)
                sliding_window_error_list.append(error_msg)
                continue

    # write sliding window errors
    with open(sliding_window_error_file, 'w') as f:
        f.writelines(sliding_window_error_list)
