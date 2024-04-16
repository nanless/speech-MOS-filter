import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm
from glob import glob
plt.switch_backend('agg')

dns_mos_threshold = 3.0
wvmos_threshold = 3.6
sigmos_threshold = 3.0
nisqamos_threshold = 4.0
utmos_thehold = 3.8

def run(statistics_folder_path, origianl_folder_path, filtered_folder_path):
    all_statistics_file_path = glob(statistics_folder_path + "/*.csv")
    df_total = pd.DataFrame()
    for statistics_file_path in tqdm(all_statistics_file_path):
        df = pd.read_csv(statistics_file_path)
        df_total = pd.concat([df_total, df], axis=0)
    df_new = pd.DataFrame(columns=df_total.columns)
    for i in tqdm(range(len(df_total))):
        filepath = df_total.iloc[i]["wav_file"]
        dnsmos_value = eval(df_total.iloc[i]["dnsmos"])["OVRL_raw"]
        wvmos_value = df_total.iloc[i]["wvmos"]
        sigmos_value = eval(df_total.iloc[i]["sigmos"])["MOS_OVRL"]
        nisqamos_value = eval(df_total.iloc[i]["nisqa"])["mos"]
        utmos_value = df_total.iloc[i]["utmos_strong"]
        if dnsmos_value >= dns_mos_threshold and wvmos_value >= wvmos_threshold and sigmos_value >= sigmos_threshold and nisqamos_value >= nisqamos_threshold and utmos_value >= utmos_thehold:
            df_new = df_new._append(df_total.iloc[i])
            out_path = filepath.replace(origianl_folder_path, filtered_folder_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copyfile(filepath, out_path)

    df_new.to_csv(statistics_folder_path+"/total_result/statistics_filtered.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--statistics_folder_path', type=str, required=True, help='The path of the statistics folder')
    parser.add_argument('--origianl_folder_path', type=str, required=True, help='The path of the original folder')
    parser.add_argument('--filtered_folder_path', type=str, required=True, help='The path of the filtered folder')
    args = parser.parse_args()
    run(args.statistics_folder_path, args.origianl_folder_path, args.filtered_folder_path)
