import os
import pandas as pd
import concurrent.futures
import random

origin_session_dir = '../traces/attack-traces/session'
new_session_dir = '../traces/attack-traces/new_session'

origin_lst = os.listdir(origin_session_dir)

NEW_MSG = ["dl_info_transfer_emm_info", "paging", "measurement_report"]
MSG = ["SIB1", "rrc_conn_req", "rrc_conn_setup", "rrc_conn_setup_complete"]

def add_msg(session):
    df = pd.read_csv(os.path.join(origin_session_dir, session))
    msg_lst = df["message name"].tolist()

    # iterate the list of messages, generate training examples
    size = len(msg_lst)
    msg1 = random.randint(0, size)
    msg2 = random.randint(0, size)
    if msg1 > msg2:
        msg1, msg2 = msg2, msg1
    df1 = df.iloc[:msg1]
    df2 = df.iloc[msg1:msg2]
    df3 = df.iloc[msg2:]
    new_msg1 = df.iloc[[msg1]].copy()
    new_msg1["message name"] = "SIB1"
    # TODO: reset non-carry on features
    new_msg2 = df.iloc[[msg1]].copy()
    new_msg2["message name"] = "SIB1"
    new_df = pd.concat([df1, new_msg1, df2, new_msg2, df3])
    new_df.to_csv(os.path.join(new_session_dir, session), index=False)

add_msg("AKA Bypass.csv")
# with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
#     for session in origin_lst:
#             executor.submit(add_msg, session)
