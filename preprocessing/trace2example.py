"""
    This script converts each trace to one example (sentence in NLP).
"""

import os
import pandas as pd
from tqdm import tqdm
import sys


def generate_one_example(csv_path, out_path):
    """convert one trace (csv file) to one training/validation example .

    Args:
        csv_path (str): path to the csv file
        out_path (str): output path of the converted trace.

    Returns:
        out (DataFrame): DataFrame of the converted trace (only one row)
    """

    df = pd.read_csv(csv_path)
    trace_length = len(df)

    if trace_length < 48: # ignore short traces
        return None 

    example = df.iloc[:, :]

    dict = {}

    for col in df.columns:
        try:
            if col == "label" or col == "EMM cause" or col == "attack_type":
                dict[col] = " ".join([str(int(e)) for e in example.loc[:, col].tolist()])
            else:
                dict[col] = " ".join([str(e) for e in example.loc[:, col].tolist()])

        except Exception as e:
            error_msg = f"{e} occured in {csv_path} for column {col}"
            print(error_msg)
            trace2example_error_list.append(error_msg)
            return None

    out = pd.DataFrame(dict, index=[0])
    
    out.to_csv(out_path, index=False)

    return out


if __name__ == "__main__":

    base_dir = r"/home/zipingye/cellular-ids"
    # base_dir = r"/Users/zipingye/Desktop/cellular-ids"

    trace2example_error_list = [] # error when convert one trace to one example (sentence in NLP)
    trace2example_error_file = os.path.join(base_dir, "preprocessing/trace2example_error.txt")

    csv_dirs = [os.path.join(base_dir, "traces/benign_traces"), os.path.join(base_dir, "traces/attack_traces")]
    out_dir = os.path.join(base_dir, "traces/train")
    os.makedirs(out_dir, exist_ok=True)

    for csv_dir in csv_dirs:

        csv_list = os.listdir(csv_dir)

        for csv_file in tqdm(csv_list):
            try:
                csv_path = os.path.join(csv_dir, csv_file)
                out_path = os.path.join(out_dir, csv_file)
                generate_one_example(csv_path, out_path)
            
            except Exception as e:
                error_msg = f"{e} occured in {csv_path} \n"
                print(error_msg)
                trace2example_error_list.append(error_msg)
                continue

    with open(trace2example_error_file, 'w') as f:
        f.writelines(trace2example_error_list)

