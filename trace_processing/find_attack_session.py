import os
import pandas as pd
import concurrent.futures

"""
    Extract sessions from CSV file
    Input: csv file
    Output: csv file with only attack session
"""
MSG = ["rrc_conn_req", "rrc_conn_setup", "rrc_conn_setup_complete_tau_req"] # indicates the end

csv_dir = '../traces/attack-traces/csv'
session_dir = '../traces/attack-traces/session'

if not os.path.exists(session_dir):
    os.mkdir(session_dir)

csv_file_lst = os.listdir(csv_dir)


def find_session(csv_file):
    
    csv_file_path = os.path.join(csv_dir, csv_file)
    df = pd.read_csv(csv_file_path)
    msg_lst = df["message name"].tolist()

    # iterate the list of messages, generate training examples
    for i in range(len(msg_lst) - 2):
        if msg_lst[i] == MSG[0]:
            if msg_lst[i+1] == MSG[1]:
                if msg_lst[i+2] == MSG[2]:
                    df.iloc[i:].to_csv(os.path.join(session_dir, csv_file), index=False)
                    break
                        

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    for csv_file in csv_file_lst:
        executor.submit(find_session, csv_file)