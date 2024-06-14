import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm
from glob import glob
import soundfile as sf
import scipy.signal as signal
import numpy as np
from multiprocessing import Pool, cpu_count

### 截止频率
cutoff_freq_default = 12000  # 最低截止频率
dns_mos_threshold_default = 3.7
wvmos_threshold_default = 3.9
sigmos_threshold_default = 3.2
nisqamos_threshold_default = 4.0
utmos_thehold_default = 3.6

def read_wav_file(filename):
    # 读取wav文件
    data, sample_rate = sf.read(filename)
    return sample_rate, data

def calculate_cutoff_frequency(filename):
    try:
        sample_rate, data = read_wav_file(filename)
    except:
        return 2000
    
    # 如果音频是立体声，则只取一个声道
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # 计算STFT
    f, t, Zxx = signal.stft(data, fs=sample_rate, nperseg=1024)
    
    # 计算频率幅度
    magnitude = np.abs(Zxx)
    
    # 对时间轴进行平均
    avg_magnitude = np.mean(magnitude, axis=1)
    
    # 找到50-2000 Hz的频率范围
    freq_range = (f >= 50) & (f <= 2000)
    
    # 计算50-2000 Hz频率范围的平均能量
    avg_energy = np.mean(avg_magnitude[freq_range])
    
    # 找到2000 Hz以上的频率
    high_freq_range = f > 2000
    
    # 找到幅度大于平均能量的5%的频率
    significant_freqs = f[high_freq_range][avg_magnitude[high_freq_range] > 0.05 * avg_energy]
    
    # 实际截止频率为这些显著频率中的最大值，如果没有则为2000
    cutoff_frequency = np.max(significant_freqs) if significant_freqs.size > 0 else 2000
    
    return cutoff_frequency

def process_file(args):
    index, row, origianl_folder_path, filtered_folder_path = args
    filepath = row["wav_file"]
    dnsmos_value = eval(row["dnsmos"])["OVRL_raw"]
    wvmos_value = row["wvmos"]
    sigmos_value = eval(row["sigmos"])["MOS_OVRL"]
    nisqamos_value = eval(row["nisqa"])["mos"]
    utmos_value = row["utmos_strong"]
    cutoff_freq = calculate_cutoff_frequency(filepath)
    row["cutoff_freq"] = cutoff_freq
    
    if (dnsmos_value >= dns_mos_threshold and wvmos_value >= wvmos_threshold and 
        sigmos_value >= sigmos_threshold and nisqamos_value >= nisqamos_threshold and 
        utmos_value >= utmos_thehold and cutoff_freq >= cutoff_freq):
        
        out_path = filepath.replace(origianl_folder_path, filtered_folder_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.copyfile(filepath, out_path)
        return row, True
    return row, False

def run(statistics_folder_paths, statisitcs_output_folder_path, origianl_folder_path, filtered_folder_path):
    all_statistics_file_path = []
    for statistics_folder_path in statistics_folder_paths:
        all_statistics_file_path += glob(statistics_folder_path + "/*.csv")
    df_total = pd.DataFrame()
    for statistics_file_path in tqdm(all_statistics_file_path):
        df = pd.read_csv(statistics_file_path)
        df_total = pd.concat([df_total, df], axis=0)
    
    df_total["cutoff_freq"] = 0
    df_total = df_total.copy()
    
    args_list = [(i, df_total.iloc[i], origianl_folder_path, filtered_folder_path) for i in range(len(df_total))]
    
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, args_list), total=len(args_list)))
    
    df_new = pd.DataFrame(columns=df_total.columns)
    for row, selected in tqdm(results):
        if selected:
            df_new = df_new._append(row)
        else:
            df_total.loc[row.name] = row

    os.makedirs(statisitcs_output_folder_path + "/total_result", exist_ok=True)
    df_total.to_csv(statisitcs_output_folder_path + "/total_result/statistics_all.csv", index=False)
    df_new.to_csv(statisitcs_output_folder_path + "/total_result/statistics_filtered.csv", index=False)

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
    parser.add_argument('--cutoff_freq', type=float, default=cutoff_freq_default, help='The default cutoff frequency')
    args = parser.parse_args()
    dns_mos_threshold = args.dns_mos_threshold
    wvmos_threshold = args.wvmos_threshold
    sigmos_threshold = args.sigmos_threshold
    nisqamos_threshold = args.nisqamos_threshold
    utmos_thehold = args.utmos_thehold
    cutoff_freq = args.cutoff_freq
    run(args.statistics_folder_paths, args.statisitcs_output_folder_path, args.origianl_folder_path, args.filtered_folder_path)
