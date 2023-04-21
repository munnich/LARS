from typing import Tuple
import numpy as np
from scipy import signal

# def enframe(x, win, inc):
#     """
#     From obspy.
#     Splits the vector up into (overlapping) frames beginning at increments
#     of inc. Each frame is multiplied by the window win().
#     The length of the frames is given by the length of the window win().
#     The centre of frame I is x((I-1)*inc+(length(win)+1)/2) for I=1,2,...
#     The mean is also subtracted from each individual frame.
#     :param x: signal to split in frames
#     :param win: window multiplied to each frame, length determines frame length
#     :param inc: increment to shift frames, in samples
#     :return f: output matrix, each frame occupies one row
#     :return length, no_win: length of each frame in samples, number of frames
#     """
#     nx = len(x)
#     nwin = len(win)
#     if (nwin == 1):
#         length = win
#     else:
#         # length = next_pow_2(nwin)
#         length = nwin
#     nf = int(np.fix((nx - length + inc) // inc))
#     # f = np.zeros((nf, length))
#     indf = inc * np.arange(nf)
#     inds = np.arange(length) + 1
#     f = x[(np.transpose(np.vstack([indf] * length)) +
#            np.vstack([inds] * nf)) - 1]
#     if (nwin > 1):
#         w = np.transpose(win)
#         f = f * np.vstack([w] * nf)
#     f = signal.detrend(f, type='constant')
#     no_win, _ = f.shape
#     return f, length, no_win

def cut_samples(data, frame_length, overlap):
    n_samples = int(np.floor(len(data) / frame_length))
    if len(data) < n_samples * frame_length + overlap:
        n_samples -= 1
    samples = [data[i * frame_length:(i + 1) * frame_length + overlap] for i in range(n_samples)]
    return np.bmat(samples)

# something here is still broken - the (working) Julia version for reference: https://gist.github.com/munnich/2d7dfc5c54fec8baa816524f72ac0294
def syllable_detection(audio, fs: int, frame_length: int = 150, overlap: int = 100, pcoeff: float = -0.97, noise_threshold = 0.05) -> list[Tuple[int, int]]:
    """
    Syllable detection according to  J. Xu, W. Liao and T. Inoue, "Speech Speed Awareness System for a Non-native Speaker," 2016 International Conference on Collaboration Technologies and Systems (CTS), Orlando, FL, USA, 2016, pp. 43-50, doi: 10.1109/CTS.2016.0027.

    :param audio: Audio array.
    :param fs: Sampling frequency.
    :param frame_length: Audio frame length for short term energy and zero crossing rate calculations.
    :param overlap: Audio frame overlap length for short term energy and zero crossing rate calculations.
    :param pcoeff: Pre-emphasis coefficient.
    :param noise_threshold: Threshold under which a syllable is considered as noise and thrown out, in samples.
    :return: List of tuples containing start and end samples for detected syllables.
    """
    syllables: list[Tuple[int, int]] = []

    # preemphasis and normalization
    audio = signal.lfilter([1, pcoeff], 1, audio)
    audio = audio / np.max(np.abs(audio))

    # short term energy
    stes = np.sum(np.abs(cut_samples(audio ** 2, frame_length, overlap)), axis=0)

    # zero crossing rate
    tmp1 = cut_samples(audio[0:-1], frame_length, overlap)
    tmp2 = cut_samples(audio[1:], frame_length, overlap)
    sgn = np.multiply(tmp1, tmp2) < 0
    zcrs = np.sum(sgn, axis=0)

    # thresholds
    ste_min = np.min(stes)
    ste_max = np.max(stes) / 4
    zcr_min = np.min(zcrs)
    zcr_max = np.max(zcrs) / 4

    N = len(zcrs)
    n = init_n = 0
    # run through all samples and check conditions
    while n < N - 1:
        if stes[n] > ste_max or zcrs[n] > zcr_max:
            n += 1
            while stes[n] > ste_min and zcrs[n] > zcr_min and n < N:
                n += 1
            # filter out noise based on length
            if (n - init_n) * (frame_length - overlap) / fs >= noise_threshold:
                syllables.append((init_n, n))
        else:
            n += 1
    return syllables

