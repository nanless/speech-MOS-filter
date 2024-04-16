import os
import soundfile as sf
import numpy as np
from mos_models.checkpoints.NISQA.nisqa.NISQA_model import nisqaModel


def get_mos(audio_path: str, output_dir_path: str):
    args = {
        "mode": "predict_file",
        "pretrained_model": "mos_models/checkpoints/NISQA/weights/nisqa.tar",
        "ms_channel": 1,
        "deg": audio_path,
        "output_dir": output_dir_path,
    }
    nisqa = nisqaModel(args)
    mos_scores = nisqa.predict()
    mos_scores = {
        "mos": mos_scores["mos_pred"].item(),
        "noi": mos_scores["noi_pred"].item(),
        "col": mos_scores["col_pred"].item(),
        "dis": mos_scores["dis_pred"].item(),
        "loud": mos_scores["loud_pred"].item(),
    }
    return mos_scores


def get_mos_scores(audio_dir_path: str, output_dir_path: str):
    args = {
        "mode": "predict_dir",
        "pretrained_model": "mos_models/checkpoints/NISQA/weights/nisqa.tar",
        "ms_channel": 1,
        "data_dir": audio_dir_path,
        "output_dir": output_dir_path,
    }
    nisqa = nisqaModel(args)
    mos_scores_df = nisqa.predict()
    filenames = [i[:-4] for i in np.array(mos_scores_df["deg"])]
    scores = np.array(mos_scores_df["mos_pred"])
    return dict(zip(filenames, scores))

def get_mos_in_dir(audio_dir_path: str, suffix=".wav"):
    args = {
        "mode": "predict_dir",
        "pretrained_model": "mos_models/checkpoints/NISQA/weights/nisqa.tar",
        "ms_channel": 1,
        "data_dir": audio_dir_path,
        "suffix": suffix,
        # "output_dir": output_dir_path,
    }
    nisqa = nisqaModel(args)
    mos_scores_df = nisqa.predict()
    return mos_scores_df

def get_mos_from_wavlist(wavlist, sr, temp_dir="./temp_nisqa"):
    os.makedirs(temp_dir, exist_ok=True)
    os.system(f"rm {temp_dir}/*")
    for i, wav_data in enumerate(wavlist):
        wav_path = f"{temp_dir}/{i}.wav"
        sf.write(wav_path, wav_data, sr)
    mos_scores_df = get_mos_in_dir(temp_dir, suffix=".wav")
    mos_scores_df['deg_numeric'] = mos_scores_df['deg'].str.extract('(\d+)').astype(int)
    mos_scores_df = mos_scores_df.sort_values(by="deg_numeric", ascending=True)
    output_list = []
    for i, row in mos_scores_df.iterrows():
        output_list.append({"mos": row["mos_pred"], "noi": row["noi_pred"], "col": row["col_pred"], "dis": row["dis_pred"], "loud": row["loud_pred"]})
    os.system(f"rm -rf {temp_dir}")
    return output_list

        

def test_get_mos():
    audio_path = "/root/data/hifitts_wav/hi_fi_tts_v0/audio/92_clean/9288/scarletletter_24_hawthorne_0204.wav"
    output_dir_path = "./temp"
    mos_score = get_mos(audio_path, output_dir_path)
    print(mos_score)

def test_get_mos_dir():
    audio_dir_path = "/mnt/d/downloads/model_E0001_B090000"
    suffix = "enhanced.wav"
    mos_scores = get_mos_in_dir(audio_dir_path, suffix)
    print(mos_scores)

def test_get_mos_from_wavlist():
    input_wavlist = []
    for i in range(500):
        input_wavlist.append(np.random.rand(16000))
    output_list = get_mos_from_wavlist(input_wavlist, 16000)
    print(output_list)

if __name__ == '__main__':
    test_get_mos_from_wavlist()