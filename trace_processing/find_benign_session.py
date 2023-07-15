import os
import pandas as pd
import concurrent.futures

"""
    Extract sessions from CSV file
    Input: csv file
    Output: csv file with only one benign session
"""

MSG1 = ["rrc_conn_req", "rrc_conn_setup", "rrc_conn_setup_complete"] # indicates the start
MSG2 = ["rrc_conn_req", "rrc_conn_setup", "rrc_conn_setup_complete_tau_req"] # indicates the end

csv_dir = '../traces/benign-traces/csv'
session_dir = '../traces/benign-traces/session'

if not os.path.exists(session_dir):
    os.mkdir(session_dir)

csv_file_lst = os.listdir(csv_dir)


def find_session(csv_file):
    
    csv_file_path = os.path.join(csv_dir, csv_file)
    df = pd.read_csv(csv_file_path)
    count = 0
    msg_lst = df["message name"].tolist()

    # iterate the list of messages, generate training examples
    for i in range(len(msg_lst) - 2):
        if msg_lst[i] == MSG1[0]:
            if msg_lst[i+1] == MSG1[1]:
                if MSG1[2] in msg_lst[i+2]:
                    start_index = i

                    for j in range(i + 1, len(msg_lst) - 3):
                        if msg_lst[j] == MSG2[0]:
                            if msg_lst[j+1] == MSG2[1]:
                                if msg_lst[j+2] == MSG2[2]:
                                    end_index = j
                                    if (end_index-start_index) > 16:
                                        df.iloc[start_index:end_index].to_csv(os.path.join(session_dir, csv_file) + str(count) + ".csv", index=False)
                                        count = count+1
                                    break
                                else:
                                    break
                        

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    for csv_file in csv_file_lst:
        executor.submit(find_session, csv_file)