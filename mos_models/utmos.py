import torch
import numpy as np
from tqdm import tqdm

def get_utmos22_strong(wave, sr):
    if type(wave) == np.ndarray:
        wave = torch.from_numpy(wave).unsqueeze(0)
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    score = predictor(wave, sr)
    return score.item()


def get_utmos22_strong_batch(waves, sr):
    if type(waves) == np.ndarray:
        waves = torch.from_numpy(waves)
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    scores = predictor(waves, sr)
    return scores.item()

def get_utmos22_strong_wavs(wavs, sr, sequential=True):
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    if torch.cuda.is_available():
        predictor = predictor.cuda()
    if sequential:
        scores = []
        for wav in tqdm(wavs):
            if type(wav) == np.ndarray:
                wav = torch.from_numpy(wav).float().unsqueeze(0)
            if torch.cuda.is_available():
                wav = wav.cuda()
            score = predictor(wav, sr)
            if torch.cuda.is_available():
                score = score.cpu()
            scores.append(score.item())
        return scores
    else:
        if type(wavs) == np.ndarray:
            wavs = torch.from_numpy(wavs).float()
        if type(wavs) == list:
            wavs = torch.from_numpy(np.stack(wavs)).float()
        scores = predictor(wavs, sr)
        return scores.tolist()
    

def test_utmos():
    waves = np.random.randn(10, 48000)
    sr = 48000
    score = get_utmos22_strong_wavs(waves, sr, sequential=True)
    print(score)
    score = get_utmos22_strong_wavs(waves, sr, sequential=False)
    print(score)

if __name__ == "__main__":
    test_utmos()