import os
import pandas as pd
import concurrent.futures

benign_session_dir = '../traces/benign-traces/csv'
attack_session_dir = '../traces/paging-traces/session'
new_trace_dir = '../traces/new-paging-traces/'

benign_lst = os.listdir(benign_session_dir)
attack_lst = os.listdir(attack_session_dir)

if not os.path.exists(new_trace_dir):
    os.mkdir(new_trace_dir)

def merge_session(benign_session, attack_session):
    benign = pd.read_csv(os.path.join(benign_session_dir, benign_session))
    attack = pd.read_csv(os.path.join(attack_session_dir, attack_session))
    msg_lst = benign["message name"].tolist()
    found = 0
    for i in range(32, len(msg_lst)):
        if msg_lst[i] == "paging":
            new_benign = benign.iloc[:i]
            found = 1
            index = i-1
            break

    if found == 0:
        return
    new_gid = new_benign["Cell ID"][index]
    new_tac = new_benign["TAC"][index]
    new_state = new_benign["EMM state"][index]
    new_substate = new_benign["EMM substate"][index]

    attack["Cell ID"] = [new_gid for _ in range(len(attack))]
    attack["TAC"] = [new_tac for _ in range(len(attack))]
    attack["EMM state"] = [new_state for _ in range(len(attack))]
    attack["EMM substate"] = [new_substate for _ in range(len(attack))]


    new_trace = pd.concat([new_benign, attack])

    new_trace.to_csv(new_trace_dir + benign_session.replace(".csv", "") + "+" + attack_session.replace(".csv", "") + ".csv", index=False)

with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    for benign_file in benign_lst:
        for attack_file in attack_lst:
            executor.submit(merge_session, benign_file, attack_file)

# for benign_file in benign_lst:
#     for attack_file in attack_lst:
#         print(benign_file)
#         print(attack_file)
#         merge_session(benign_file, attack_file)