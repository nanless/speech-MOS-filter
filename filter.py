import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm
from glob import glob
plt.switch_backend('agg')

dns_mos_threshold_default = 3.5
wvmos_threshold_default = 3.8
sigmos_threshold_default = 3.0
nisqamos_threshold_default = 4.1
utmos_thehold_default = 3.5

def run(statistics_folder_paths, statisitcs_output_folder_path, origianl_folder_path, filtered_folder_path):
    all_statistics_file_path = []
    for statistics_folder_path in statistics_folder_paths:
        all_statistics_file_path += glob(statistics_folder_path + "/*.csv")
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
    os.makedirs(statisitcs_output_folder_path+"/total_result", exist_ok=True)
    df_new.to_csv(statisitcs_output_folder_path+"/total_result/statistics_filtered.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--statistics_folder_paths', type=str, required=True, default=None, nargs='+', help='The paths of the statistics folders')
    parser.add_argument('--statisitcs_output_folder_path', type=str, required=True, help='The path of the statistics output folder')
    parser.add_argument('--origianl_folder_path', type=str, required=True, help='The path of the original folder')
    parser.add_argument('--filtered_folder_path', type=str, required=True, help='The path of the filtered folder')
    parser.add_argument('--dns_mos_threshold', type=float, default=dns_mos_threshold_default, help='The threshold of DNS MOS')
    parser.add_argument('--wvmos_threshold', type=float, default=wvmos_threshold_default, help='The threshold of WV MOS')
    parser.add_argument('--sigmos_threshold', type=float, default=sigmos_threshold_default, help='The threshold of SIGMOS')
    parser.add_argument('--nisqamos_threshold', type=float, default=nisqamos_threshold_default, help='The threshold of NISQ MOS')
    parser.add_argument('--utmos_thehold', type=float, default=utmos_thehold_default, help='The threshold of UT MOS')
    args = parser.parse_args()
    dns_mos_threshold = args.dns_mos_threshold
    wvmos_threshold = args.wvmos_threshold
    sigmos_threshold = args.sigmos_threshold
    nisqamos_threshold = args.nisqamos_threshold
    utmos_thehold = args.utmos_thehold
    run(args.statistics_folder_paths, args.statisitcs_output_folder_path, args.origianl_folder_path, args.filtered_folder_path)
