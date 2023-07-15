#!/usr/bin/python
# Filename: offline-analysis-example.py
import os
import concurrent.futures
from tqdm import tqdm

"""
    Offline analysis by replaying logs
    Input: mi2log file
    Output: csv file
"""

# Import MobileInsight modules
from lte_analyzer_spy import LteAnalyzerSpy
from mobile_insight.monitor import OfflineReplayer


mi2log_dir = "../../traces/"
csv_dir = "../../csv"

mi2log_lst = os.listdir(mi2log_dir)
os.makedirs(csv_dir, exist_ok=True)


def tocsv(mi2log_file): 
    # Initialize a 4G monitor
    src = OfflineReplayer()
    input_file = os.path.join(mi2log_dir, mi2log_file)
    output_file = os.path.join(csv_dir, mi2log_file.replace("mi2log", "csv"))
    src.set_input_path(input_file)

    rrc_analyzer = LteAnalyzerSpy(output_file)
    rrc_analyzer.set_source(src)
    
    # Start the monitoring
    src.run()
    rrc_analyzer.toCSV()
    rrc_analyzer.reset()


if __name__ == "__main__":

    for mi2log_file in tqdm(mi2log_lst):

        try:
            tocsv(mi2log_file)

        except TypeError as te:
            continue

        except Exception as e:
            print(mi2log_file, e)
            continue
            # f = open("./error.txt", "a")
            # f.write(mi2log_file)
            # f.write("\n")
            # f.close()

# with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
#     executor.map(tocsv, mi2log_lst)
#     # for mi2log_file in mi2log_lst:
#     #     executor.submit(tocsv, mi2log_file)
        