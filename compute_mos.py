import os
import soundfile as sf
from glob import glob
from metrics import DNSMOS, WVMOS, SIGMOS, NISQA, UTMOS_STRONG
import pandas as pd
import sys
import argparse
import librosa
import datetime

def strip_silence_head_tail(wav, sr):
    # Remove silence at the beginning and end of the audio signal
    non_silence_indices = librosa.effects.split(wav, top_db=40)
    if len(non_silence_indices) == 0:
        return wav
    start_idx = non_silence_indices[0][0]
    end_idx = non_silence_indices[-1][1]
    wav_new = wav[start_idx:end_idx]
    sf.write("temp_nostripped.wav", wav, sr)
    sf.write("temp_stripped.wav", wav_new, sr)
    return wav_new

def downsample_to_24k(wav, sr):
    # Downsample to 24k
    if sr!= 24000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=24000, res_type='fft')
    return wav, 24000

def normalize_signal(wav):
    # Normalize the signal to have max value of 0.9
    max_val = max(abs(wav))
    gain = 0.9 / max_val
    wav = wav * gain
    return wav

# New function to process each batch
def process_batch(batch_wav_files, batch_index, strip_silence=False, down24k=False, normalize=False, save_wavs=False, temp_folder_postfix=""):

    output_statistics_file = os.path.join(output_folder, f"statistics_{batch_index}.csv")

    batch_wavs = []
    for wav_file in batch_wav_files:
        try:
            wav, sr = sf.read(wav_file)
            if len(wav.shape) > 1:
                wav = wav[:, 0]
            if strip_silence:
                wav = strip_silence_head_tail(wav, sr)
            if down24k:
                wav, sr = downsample_to_24k(wav, sr)
            if normalize:
                wav = normalize_signal(wav)
            if save_wavs:
                if not os.path.exists(output_folder+"/output_wavs"):
                    os.makedirs(output_folder+"/output_wavs")
                output_path = wav_file.replace(source_folder, output_folder+"/output_wavs")
                dir_path = os.path.dirname(output_path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                sf.write(output_path, wav, sr)
            wav_seconds = len(wav)/sr
            if 0.6 < wav_seconds < 88:
                print(f"Added {wav_file} wav length in seconds: {wav_seconds}")
                batch_wavs.append(wav)
            else:
                print(f"Skipping {wav_file} wav length in seconds: {wav_seconds}")
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue
    
    nisqa_dir = f"temp_nisqa_{batch_index}"
    nisqa_dir = nisqa_dir + "_" + temp_folder_postfix + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Your existing code to calculate MOS scores
    print(f"Processing batch {batch_index}...")
    print("Calculating DNSMOS scores")
    try:
        dnsmos_scores = DNSMOS(batch_wavs, sr)
    except Exception as e:
        print(f"Error calculating DNSMOS scores: {e}")
        dnsmos_scores = []
    print("Calculating WVMOS scores")
    try:
        wvmos_scores = WVMOS(batch_wavs, sr)
    except Exception as e:
        print(f"Error calculating WVMOS scores: {e}")
        wvmos_scores = []        
    print("Calculating SIGMOS scores")
    try:
        sigmos_scores = SIGMOS(batch_wavs, sr)
    except Exception as e:
        print(f"Error calculating SIGMOS scores: {e}")
        sigmos_scores = []
    print("Calculating NISQA scores")
    try:
        nisqa_scores = NISQA(batch_wavs, sr, nisqa_dir)
    except Exception as e:
        print(f"Error calculating NISQA scores: {e}")
        nisqa_scores = []
    print("Calculating UTMOS scores")
    try:
        utmos_scores = UTMOS_STRONG(batch_wavs, sr)
    except Exception as e:
        print(f"Error calculating UTMOS scores: {e}")
        utmos_scores = []
    
    if len(dnsmos_scores) != len(batch_wavs) or len(wvmos_scores) != len(batch_wavs) or len(sigmos_scores) != len(batch_wavs) or len(nisqa_scores) != len(batch_wavs) or len(utmos_scores) != len(batch_wavs):
        print(f"Error: lengths of scores do not match: {len(dnsmos_scores)}, {len(wvmos_scores)}, {len(sigmos_scores)}, {len(nisqa_scores)}, {len(utmos_scores)}")
        return 1

    results = []
    for j in range(len(batch_wavs)):
        dnsmos_score = {"OVRL_raw": dnsmos_scores[j]["OVRL_raw"], "SIG_raw": dnsmos_scores[j]["SIG_raw"]}
        results.append({"wav_file": batch_wav_files[j], "dnsmos": dnsmos_score, "wvmos": wvmos_scores[j], "sigmos": sigmos_scores[j], "nisqa": nisqa_scores[j], "utmos_strong": utmos_scores[j]})

    # Convert results to DataFrame and save
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_statistics_file, index=False)
    return 0

# Splitting wav_files into chunks
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def filterout_previous_statistics(wav_files, previous_statistics_dirs):
    if previous_statistics_dirs is None:
        return wav_files
    previous_statistics_wav_files = set()
    for previous_statistics_dir in previous_statistics_dirs:
        previous_statistics_files = glob(f"{previous_statistics_dir}/*.csv")
        for file in previous_statistics_files:
            df = pd.read_csv(file)
            previous_statistics_wav_files.update(df["wav_file"].tolist())
    filtered_wav_files = [file for file in wav_files if file not in previous_statistics_wav_files]
    return filtered_wav_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_files_dir', type=str, help='Directory of wav files')
    parser.add_argument('--output_statistics_dir', type=str, help='Directory to save statistics')
    # add an optional argument for previous statistics directories (maybe a list)
    parser.add_argument('--previous_statistics_dirs', type=str, default=None, nargs='+', help='Directories of previous statistics to skip processing')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for multiprocessing')
    parser.add_argument('--strip_silence', action='store_true', help='Whether to strip silence or not')
    parser.add_argument('--down24k', action='store_true', help='Whether to downsample to 24k or not')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize the signal or not')
    parser.add_argument('--save_wavs', action='store_true', help='Whether to save the processed wavs or not')
    parser.add_argument('--temp_folder_postfix', type=str, default="", help='Postfix for temporary folder for NISQA')
    args = parser.parse_args()
    source_folder = args.wav_files_dir
    output_folder = args.output_statistics_dir
    previous_statistics_dirs = args.previous_statistics_dirs
    strip_silence = args.strip_silence
    down24k = args.down24k
    normalize = args.normalize
    batch_size = args.batch_size
    save_wavs = args.save_wavs
    temp_folder_postfix = args.temp_folder_postfix

    if not os.path.exists(source_folder):
        print(f"Source folder {source_folder} does not exist")
        sys.exit(1)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def walk_folder_wavfiles(folder):
        wav_files = glob(f"{folder}/**/*.wav", recursive=True)
        return wav_files

    wav_files = walk_folder_wavfiles(source_folder)
    print(f"Total {len(wav_files)} wav files")
    filtered_wav_files = filterout_previous_statistics(wav_files, previous_statistics_dirs)
    print(f"Filtered to {len(filtered_wav_files)} wav files")
    wav_files = filtered_wav_files

    chunked_wav_files = list(chunks(wav_files, batch_size))

    for i, chunk in enumerate(chunked_wav_files):
        print(f"Processing chunk {i}")
        process_batch(chunk, i, strip_silence, down24k, normalize, save_wavs, temp_folder_postfix)