import numpy as np
import librosa
from mos_models.dnsmos_local import compute_dnsmos
from mos_models.wv_mos import get_wvmos
from mos_models.sigmos import SigMOS
from mos_models.nisqa import get_mos_from_wavlist
from mos_models.utmos import get_utmos22_strong_wavs
import datetime
import multiprocessing
from functools import partial

def DNSMOS(est, sr=16000):
    return compute_dnsmos(est, sr)

def WVMOS(ests, sr=16000):
    model = get_wvmos()
    mos = model.calculate_wavs(ests, sr)
    return mos

def run_sigmos(est, sr, sigmos_path):
    mode = SigMOS(sigmos_path)  # Create a new instance of SigMOS in each process
    return mode.run(est, sr)

def SIGMOS(ests, sr=48000, sigmos_path="mos_models/checkpoints/SIGMOS"):
    # Create a pool of processes
    with multiprocessing.Pool() as pool:
        # Use a partial function with the shared arguments
        func = partial(run_sigmos, sr=sr, sigmos_path=sigmos_path)

        # Map the ests to the function across the pool
        mos_list = pool.map(func, ests)

    return mos_list

###temp_dir 用datatime时间戳命名
def NISQA(ests, sr=48000, temp_dir="temp_nisqa_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")):
    return get_mos_from_wavlist(ests, sr, temp_dir)

def UTMOS_STRONG(ests, sr=48000):
    return get_utmos22_strong_wavs(ests, sr)
        

# Only registered metric can be used.
REGISTERED_METRICS = {
    "DNSMOS": DNSMOS,
    "WVMOS": WVMOS,
    "SIGMOS": SIGMOS,
    "NISQA": NISQA,
    "UTMOS_STRONG": UTMOS_STRONG,
}
