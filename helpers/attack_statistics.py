
import pandas as pd
import os
import sys
from tqdm import tqdm
import json
import argparse
import random


if __name__ == "__main__":

    base_dir = "/home/zipingye/cellular-ids"

    attack_names = ["AKA_Bypass", "Attach_Reject", "EMM_Information", "IMEI_Catching",
                    "IMSI_Catching", "Malformed_Identity_Request", "Null_Encryption",
                    "Numb_Attack", "Repeat_Security_Mode_Command", "RLF_Report",
                    "Service_Reject", "TAU_Reject"]

    attack_stat = dict.fromkeys(attack_names, 0)

    trace_list = ["exclude_0_attacks_version_1"] + [f"exclude_2_attacks_version_{i}" for i in range(1,7)]

    for trace in trace_list:
        print(trace)
        val_df = pd.read_csv(os.path.join(base_dir, "traces", trace, "validation.csv"))
        all_attack_types = [int(at) for seq in val_df["attack_type"] for at in seq.split()]
 
        for i in range(len(attack_names)):
            attack_stat[attack_names[i]] = all_attack_types.count(i)

        print(attack_stat)

        with open(os.path.join(base_dir, "traces", trace, "val_attack_stat.txt"), 'w') as f:
            for k, v in attack_stat.items():
                f.write(f"{k} : {v}")

