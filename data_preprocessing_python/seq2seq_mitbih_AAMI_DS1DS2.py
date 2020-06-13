import wfdb
import os
import numpy as np
from scipy import signal
from tqdm import tqdm

_data_dir = os.path.join(os.path.expanduser("~"), "dataset/ECG/mitbih")

AAMI_annotations = {'N' 'S' 'V' 'F' 'Q'}
AAMI2_annotations = {'N' 'S' 'V_hat' 'Q'}
annots_list = ['N', 'L', 'R', 'e', 'j', 'S', 'A', 'a', 'J', 'V', 'E', 'F', '/', 'f', 'Q']

N_g = ['N', 'L', 'R', 'e', 'j']  # 0
S_g = ['A', 'a', 'J', 'S']  # 1
V_g = ['V', 'E']  # 2
F_g = ['F']  # 3
Q_g = [' /', 'f', 'Q']  # 4

AAMI2MITBIH_MAPPING = {
    "N": ['N', 'L', 'R', 'e', 'j'],
    "S": ['A', 'a', 'J', 'S'],
    "V": ['V', 'E'],
    "F": ['F'],
    "Q": [' /', 'f', 'Q'],
}

AAMI2MITBIH_MAPPING2 = {
    "N": ['N', 'L', 'R'],
    "S": ['A', 'a', 'J', 'S', 'e', 'j'],
    "V": ['V', 'E'],
    "F": ['F'],
    "Q": ['P', '/', 'f'],
}
annots_list2 = ['N', 'L', 'R', 'e', 'j', 'S', 'A', 'a', 'J', 'V', 'E', 'F', '/', 'f', 'P']

DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
index = 1
beat_len = 280
n_cycles = 0


def normalize(values):
    mean = np.mean(values)
    std = np.std(values)
    return [(v - mean) / std for v in values]


def qrs_onoffset(search_interval, mode):
    slope = [abs(search_interval[i + 2] - search_interval[i]) for i in range(len(search_interval) - 2)]
    if mode == "on":
        index = np.argmin(slope)
    else:
        slope_th = 0.2 * max(slope)
        index = np.where(np.array(slope) >= slope_th)[0]
    return index


def all_markers(ecg, r_peaks, fs, remove_invalid=True):
    """
    all 7 markers around the r_peak, including
    [P-wave peak, QRS on, Q, R-peak, S, QRS off, T-wave peak]
    [p, qrs_start, q, r, s, qrs_end, t]

    the duration of QRS complex varies from 0.1s to 0.2s

    :param ecg: List[float] or 1-dim np.array.
    :param r_peaks: List[int]
    :param fs: int
    :param remove_invalid: bool
    :return: List[List[int]], list of list of all markers
    """

    # average of number of samples in a beat
    aveHB = len(ecg) // len(r_peaks)

    # setup search windows
    windowQ = round(fs * 0.05)
    windowS = round(fs * 0.1)
    windowOF = round(fs * 0.04)

    windowP = round(aveHB / 3)
    windowT = round(aveHB * 2 / 3)

    fid_pks = np.zeros((len(r_peaks), 7))

    qrs_start = 0
    q = 0
    s = 0
    qrs_end = 0
    for i in range(len(r_peaks)):
        r = r_peaks[i]
        if i == 0:
            qrs_end = r + windowS
        elif i == len(r_peaks) - 1:
            qrs_start = r - windowQ
        elif r + windowT < len(ecg) and r - windowP >= 0:
            q = r - windowQ + np.argmin(ecg[r - windowQ:r])
            s = r + np.argmin(ecg[r:r + windowS])
            qrs_start = r - windowOF + qrs_onoffset(ecg[r - windowOF:r], "on")
            qrs_end = r + qrs_onoffset(ecg[r:r + windowOF], "off")

        fid_pks[i, :] = [0, qrs_start, q, r, s, qrs_end, 0]

    for i in range(1, len(r_peaks) - 1):
        this_on = fid_pks[i, 1]  # this qrs_start
        next_on = fid_pks[i + 1, 1]  # next qrs_start
        last_off = fid_pks[i - 1, 5]  # last qrs_end
        this_off = fid_pks[i, 5]  # this qrs_end

        if last_off < this_on and this_off < next_on:
            search_last = round((1 / 3) * (this_on - last_off))
            # p
            fid_pks[i, 0] = (this_on - search_last) + max(ecg[(this_on - search_last): this_on])

            search_next = round((2 / 3) * (next_on - this_off))
            # t
            fid_pks[i, 6] = this_off + max(ecg[this_off: this_off + search_next])

    if remove_invalid:
        # validate the data, make sure no zero exists
        fid_pks_valid = [pks for pks in fid_pks if np.all(np.array(pks) != 0)]
        return fid_pks_valid
    else:
        return fid_pks


def record2beats(record_id):
    """

    :param record_id: str
    :return:
    """
    lead_id = 0

    record_id = str(record_id)
    sig, field = wfdb.rdsamp(os.path.join(_data_dir, record_id))
    ann = wfdb.rdann(os.path.join(_data_dir, record_id), extension='atr')

    sig_freq = field["fs"]
    rPeaks = ann.sample
    all_peaks = all_markers(sig[lead_id], rPeaks, sig_freq, remove_invalid=False)

    valid_indices = np.array([np.all(np.array(pks) != 0) for pks in all_peaks])
    peaks = np.array(all_peaks[valid_indices])
    annot = np.array(ann.symbol[valid_indices])

    sig_normalized = normalize(sig[lead_id])

    seg_values = []
    seg_labels = []
    for j in range(len(annot)):

        # find aami_label for annot[j]
        if annot[j] not in annots_list:
            continue
        else:
            label = None
            for aami_label in AAMI2MITBIH_MAPPING:
                if annot[j] in AAMI2MITBIH_MAPPING[aami_label]:
                    label = aami_label
                    break

            if label is None:
                raise Exception("no such label")

        if j == 0:
            # this is probabily wrong
            original_beat = sig_normalized[:min(sig_freq, peaks[j, -1])]
        else:
            original_beat = sig_normalized[peaks[j - 1, -1]:peaks[j, -1]]

        seg_values.append(signal.resample(original_beat, beat_len))
        seg_labels.append(annot[j])

    # strange reformation to keep consistency with MATLAB
    seg_values = [np.reshape(beat, (1, len(beat), 1)) for beat in seg_values]
    # dim: beat, unknown?, amplitude, channel

    seg_labels = ["".join(seg_labels)]
    # dim: unknown?, beat

    assert sum([len(beat[0]) for beat in seg_values]) == len(seg_labels[0])

    return seg_values, seg_labels


def ds2beats(DS):
    """

    :param DS: List[int]
    :return: List[List[float]], List[str], list of beats and list of AAMI labels
    """

    seg_values = []
    seg_labels = []
    for record_id in tqdm(DS):
        values, labels = record2beats(str(record_id))
        seg_values.append(values)
        seg_labels.append(labels)

    return seg_values, seg_labels


if __name__ == "__main__":
    dataset = {
        "s2s_mitbih_DS1": ds2beats(DS1),
        "s2s_mitbih_DS2": ds2beats(DS2),
    }
