import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
plt.switch_backend('agg')

def run(statistics_folder_path):

    all_statistics_file_path = glob(statistics_folder_path + "/*.csv")
    df_total = pd.DataFrame()
    for statistics_file_path in tqdm(all_statistics_file_path):
        df = pd.read_csv(statistics_file_path)
        df_total = pd.concat([df_total, df], axis=0)

    os.makedirs(statistics_folder_path+"/total_result", exist_ok=True)
    dnsmos_values = []
    wvmos_values = []
    sigmos_values = []
    nisqamos_values = []
    utmos_values = []
    for i in tqdm(range(len(df_total))):
        dnsmos_values.append(eval(df_total.iloc[i]["dnsmos"])["OVRL_raw"])
        wvmos_values.append(df_total.iloc[i]["wvmos"])
        sigmos_values.append(eval(df_total.iloc[i]["sigmos"])["MOS_OVRL"])
        nisqamos_values.append(eval(df_total.iloc[i]["nisqa"])["mos"])
        utmos_values.append(df_total.iloc[i]["utmos_strong"])

    ### plot histgram of dnsmos
    plt.figure(figsize=(20, 20))
    plt.hist(dnsmos_values, bins=200, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xticks(np.arange(1, 5.1, 0.1))
    plt.xlabel("dnsmos")
    plt.ylabel("density")
    plt.title("histgram of dnsmos")
    plt.savefig(statistics_folder_path+"/total_result/histgram_dnsmos.png")


    ### plot histgram of wvmos
    plt.figure(figsize=(20, 20))
    plt.hist(wvmos_values, bins=200, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.xticks(np.arange(0, 6, 0.1))
    plt.xlabel("wvmos")
    plt.ylabel("density")
    plt.title("histgram of wvmos")
    plt.savefig(statistics_folder_path+"/total_result/histgram_wvmos.png")

    ### plot histgram of sigmos
    plt.figure(figsize=(20, 20))
    plt.hist(sigmos_values, bins=200, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xticks(np.arange(0, 5.2, 0.1))
    plt.xlabel("sigmos")
    plt.ylabel("density")
    plt.title("histgram of sigmos")
    plt.savefig(statistics_folder_path+"/total_result/histgram_sigmos.png")

    ### plot histgram of nisqamos
    plt.figure(figsize=(20, 20))
    plt.hist(nisqamos_values, bins=200, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xticks(np.arange(0, 6, 0.1))
    plt.xlabel("nisqamos")
    plt.ylabel("density")
    plt.title("histgram of nisqamos")
    plt.savefig(statistics_folder_path+"/total_result/histgram_nisqamos.png")

    ### plot histgram of utmos
    plt.figure(figsize=(20, 20))
    plt.hist(utmos_values, bins=200, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xticks(np.arange(0, 5, 0.1))
    plt.xlabel("utmos")
    plt.ylabel("density")
    plt.title("histgram of utmos")
    plt.savefig(statistics_folder_path+"/total_result/histgram_utmos.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--statistics_folder_path', type=str, required=True, help='path to the folder containing all statistics files')
    args = parser.parse_args()
    run(args.statistics_folder_path)