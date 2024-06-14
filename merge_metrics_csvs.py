import glob
import pandas as pd
import os

dest_folder = "/root/autodl-tmp/DNS_challenge5_data/total_mos_results/unfiltered"
original_paths = glob.glob("/root/autodl-tmp/DNS_challenge5_data/clean_mos_*/*.csv")

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

df_total = pd.DataFrame()
for path in original_paths:
    df = pd.read_csv(path)
    df_total = pd.concat([df_total, df], axis=0)

df_total.to_csv(os.path.join(dest_folder, "total_mos_results.csv"), index=False)