import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

def find_all_EMM_states(df):
    EMM_states = []

    for _, row in df.iterrows():
        current_EMM_state = row.loc["EMM state"]
        
        if current_EMM_state not in EMM_states:
            EMM_states.append(current_EMM_state)

    return list(set(EMM_states))

def find_all_EMM_substates(df):
    EMM_substates = []

    for _, row in df.iterrows():
        current_EMM_substate = row.loc["EMM substate"]
        
        if current_EMM_substate not in EMM_substates:
            EMM_substates.append(current_EMM_substate)

    return list(set(EMM_substates))

'''
def find_all_msg_name(df):
    message_names = []

    for _, row in df.iterrows():
        current_msg_name = row.loc["message name"]
        
        if current_msg_name not in message_names:
            message_names.append(current_msg_name)

    return list(set(message_names))
'''

if __name__ == "__main__":
    csv_dir = '../traces'
    csv_lst = os.listdir(csv_dir)

    df = pd.DataFrame()
    for csv_file in tqdm(csv_lst):
        csv_path = os.path.join(csv_dir, csv_file)
        df = pd.concat([df, pd.read_csv(csv_path).iloc[: , 1:]])

    print("total number of messages:", len(df))

    EMM_states = find_all_EMM_states(df)
    with open("EMM_states.txt", 'w') as f:
        for EMM_state in EMM_states:
            f.write(f"{EMM_state}\n")
        f.write(f"{len(EMM_states)}")

    EMM_substates = find_all_EMM_substates(df)
    with open("EMM_substates.txt", 'w') as f:
        for EMM_substate in EMM_substates:
            f.write(f"{EMM_substate}\n")
        f.write(f"{len(EMM_substates)}")

    # message_names = find_all_msg_name(df)
    # with open("message_names.txt", 'w') as f:
    #     for message_name in message_names:
    #         f.write(f"{message_name}\n")
    #     f.write(f"{len(message_names)}")

    print("EMM states", EMM_states)
    print("total number of EMM states:", len(EMM_states))
    print("EMM substates", EMM_substates)
    print("total number of EMM substates:", len(EMM_substates))
    # print("message names", message_names)
    # print("total number of message names:", len(message_names))
