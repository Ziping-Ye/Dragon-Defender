
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

def find_reject(message_name, csv_path):

    df = pd.read_csv(csv_path)

    # fill 0 if EMM cause is empty
    df.loc[:, "EMM cause"].replace(np.NaN, 0, inplace=True)

    msg_name_lst = df.loc[:, "message name"].tolist()
    EMM_cause_lst = df.loc[:, "EMM cause"].tolist()

    msg_EMM_dict = dict(zip(msg_name_lst, EMM_cause_lst))
    
    assert len(msg_name_lst) == len(EMM_cause_lst), "length of message name and EMM cause should be the same"

    for msg_name, _ in zip(msg_name_lst, EMM_cause_lst):
        if message_name == msg_name:
            msg_EMM_cause.append(msg_EMM_dict[msg_name])


if __name__ == "__main__":

    message_names = ["dl_info_transfer_service_reject", "dl_info_transfer_tau_reject", "dl_info_transfer_attach_reject"]

    csv_dir = "./traces/benign-traces/csv"
    csv_lst = os.listdir(csv_dir)

    msg_name_EMM_cause_list = []

    for message_n in message_names:

        msg_EMM_cause = []

        for csv_file in tqdm(csv_lst):
            csv_path = os.path.join(csv_dir, csv_file)
            find_reject(message_n, csv_path)

        plt.hist(msg_EMM_cause)
        plt.savefig(f"{message_n}_EMM_cause_hist.png")

        msg_EMM_cause = [int(cause) for cause in msg_EMM_cause]

        with open(f"./{message_n}_EMM_cause.txt", 'w') as f:
            f.write("\n".join([str(cause) for cause in msg_EMM_cause]))
        
        print(f"set of {message_n}: ")
        print("set: ", set(msg_EMM_cause))
        print("Counter: ", Counter(list(msg_EMM_cause)))
