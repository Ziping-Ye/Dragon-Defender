import os
import pandas as pd
import concurrent.futures

benign_session_dir = '../traces/benign-traces/session'
attack_session_dir = '../traces/attack-traces/session'
new_trace_dir = '../traces/new-attack-traces/'

benign_lst = os.listdir(benign_session_dir)
attack_lst = os.listdir(attack_session_dir)

if not os.path.exists(new_trace_dir):
    os.mkdir(new_trace_dir)

def merge_session(benign_session, attack_session):
    benign = pd.read_csv(os.path.join(benign_session_dir, benign_session))
    attack = pd.read_csv(os.path.join(attack_session_dir, attack_session))
    new_gid = attack["Cell ID"][0]
    new_tac = attack["TAC"][0]
    new_trace = pd.concat([benign, attack])
    concat_index = len(benign)
    for i in range(concat_index, 0, -1):
        if new_trace.iloc[i]["message name"] == "SIB1":
            for j in range(i, concat_index):
                new_trace.at[j, "Cell ID"] = new_gid
                new_trace.at[j, "TAC"] = new_tac
            break
    new_trace.to_csv(new_trace_dir + benign_session.replace(".csv", "") + "+" + attack_session.replace(".csv", "") + ".csv", index=False)

with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    for benign_file in benign_lst:
        for attack_file in attack_lst:
            executor.submit(merge_session, benign_file, attack_file)