import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.signal import resample
import neurokit2 as nk

def signal_to_ptt(ecg, ppg, fs=125, ptt_min=0.05, ptt_max=0.5, beat_len=128):
    # ppt = t_(ppg peak) - t_(ecg r-peak)

    # clean ecg signal and isolate r-peaks to one column
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs)
    r_peaks_info = nk.ecg_findpeaks(ecg_clean, sampling_rate=fs)
    r_peaks = r_peaks_info["ECG_R_Peaks"]

    # clean ppg signal and isolate peaks to one column 
    ppg_clean = nk.ppg_clean(ppg, sampling_rate=fs)
    ppg_peaks_info = nk.ppg_findpeaks(ppg_clean, sampling_rate=fs)
    ppg_peaks = ppg_peaks_info["PPG_Peaks"]

    beats = []
    search_window = int(0.3 * fs)

    # find peak onset for CNN
    for ppg_peak in ppg_peaks: # ppg_peak refers to idx of peak
        start = max(0, ppg_peak - search_window) # start index, either 0 or index after the nth window
        if start >= ppg_peak: # if start greater than ppg_peak idx
            continue
        window = ppg_clean[start:ppg_peak] # create window 
        if len(window) == 0:
            continue
        onset_idx = start + np.argmin(window) # onset index

        # Find nearest preceding R-peak before PPG peak for PTT
        preceding_r = r_peaks[r_peaks < ppg_peak] # boolean mask, conv. to array of all r_peaks < ppg_peaks
        if len(preceding_r) == 0:
            continue
        r_peak = preceding_r[-1] # preceding_r is sorted, find last array (i.e. greatest r-peak before ppg peak)

        ptt = (ppg_peak - r_peak) / fs
        if not (ptt_min <= ptt <= ptt_max):
            continue

        # Find end onset 
        next_ppg_peaks = ppg_peaks[ppg_peaks > ppg_peak] # returns every future beat peak
        end_idx = onset_idx + int(1.2 * fs)  # generous fallback window; hard cutoff
        if len(next_ppg_peaks) > 0:
            next_start_search = max(ppg_peak, next_ppg_peaks[0] - search_window)
            end_idx = min(end_idx, next_start_search + 1)

        raw_beat = ppg_clean[onset_idx:end_idx] # isolate beat
        if len(raw_beat) < 10:
            continue

        norm_beat = resample(raw_beat, beat_len) # normalize to 128 samples; fourier-based resampling
        # min-max normalization (0-1 normalization)
        norm_beat = (norm_beat - norm_beat.min()) / (norm_beat.max() - norm_beat.min() + 1e-8)

        # append all collected data to beats
        beats.append({
            "ppg_beat": norm_beat.astype(np.float32),
            "ptt": ptt,
            "r_peak_idx": r_peak,
            "ppg_peak_idx": ppg_peak,
        })

    return beats, r_peaks

