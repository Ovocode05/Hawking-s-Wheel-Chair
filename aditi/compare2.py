import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import welch, cwt, ricker
from scipy.stats import entropy
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


# ================================================================
# LOAD SEGMENTS
# ================================================================
def load_segments(path):
    data = np.load(path, allow_pickle=True)
    segs_raw = data["segments"]

    cleaned = []
    for seg in segs_raw:
        arr = np.array(seg, dtype=float)
        cleaned.append(arr)

    print(f"Loaded {len(cleaned)} segments from {path}")
    return cleaned



# ================================================================
# METRIC 1 — DTW Distance (full 5-D)
# ================================================================
def dtw_distance(segA, segB):
    dist, _ = fastdtw(segA, segB, dist=euclidean)
    return dist



# ================================================================
# METRIC 2 — FFT Power Spectrum Distance
# ================================================================
def fft_distance(sigA, sigB):
    fA, pA = welch(sigA)
    fB, pB = welch(sigB)
    L = min(len(pA), len(pB))
    return np.sum((pA[:L] - pB[:L])**2)



# ================================================================
# METRIC 3 — Dominant Frequency Difference
# ================================================================
def dominant_freq_distance(sigA, sigB):
    fA, pA = welch(sigA)
    fB, pB = welch(sigB)
    return abs(fA[np.argmax(pA)] - fB[np.argmax(pB)])



# ================================================================
# METRIC 4 — Spectral Entropy Distance
# ================================================================
def spectral_entropy_dist(sigA, sigB):
    _, pA = welch(sigA)
    _, pB = welch(sigB)
    pA /= pA.sum()
    pB /= pB.sum()
    return abs(entropy(pA) - entropy(pB))



# ================================================================
# METRIC 5 — Wavelet Transform Distance
# ================================================================
def wavelet_distance(sigA, sigB):
    widths = np.arange(1, 31)
    cA = cwt(sigA, ricker, widths)
    cB = cwt(sigB, ricker, widths)
    L = min(cA.shape[1], cB.shape[1])
    return np.linalg.norm(cA[:, :L] - cB[:, :L])



# ================================================================
# METRIC 6 — Trend Distance (slope difference)
# ================================================================
def trend_distance(sigA, sigB):
    tA = np.arange(len(sigA))
    tB = np.arange(len(sigB))
    slopeA = np.polyfit(tA, sigA, 1)[0]
    slopeB = np.polyfit(tB, sigB, 1)[0]
    return abs(slopeA - slopeB)



# ================================================================
# METRIC 7 — Seasonality Strength Distance (STL)
# ================================================================
def seasonality_distance(sigA, sigB):
    try:
        A = STL(sigA, period=20, robust=True).fit()
        B = STL(sigB, period=20, robust=True).fit()
        return abs(np.var(A.seasonal) - np.var(B.seasonal))
    except:
        return 0.0   # fallback for very-short segments



# ================================================================
# METRIC 8 — Stationarity (ADF Statistic Difference)
# ================================================================
def stationarity_distance(sigA, sigB):
    try:
        adfA = adfuller(sigA)[0]
        adfB = adfuller(sigB)[0]
        return abs(adfA - adfB)
    except:
        return 0.0



# ================================================================
# METRIC 9 — Volatility Distance (rolling std)
# ================================================================
def volatility_distance(sigA, sigB, win=10):
    volA = np.std(sigA[:win])
    volB = np.std(sigB[:win])
    return abs(volA - volB)



# ================================================================
# WORD-LEVEL ALL-METRIC COMPARISON
# ================================================================
def compare_words(wordA, wordB, nameA="A", nameB="B"):

    print(f"\n\n===============================================")
    print(f"      COMPARING WORD '{nameA}' vs '{nameB}'")
    print(f"===============================================\n")

    metric_names = [
        "DTW",
        "FFT",
        "DominantFreq",
        "SpectralEntropy",
        "Wavelet",
        "Trend",
        "Seasonality",
        "Stationarity",
        "Volatility"
    ]

    results = {m: [] for m in metric_names}

    for i, segA in enumerate(wordA):
        for j, segB in enumerate(wordB):

            thetaA = segA[:,0]
            thetaB = segB[:,0]

            m_dtw = dtw_distance(segA, segB)
            m_fft = fft_distance(thetaA, thetaB)
            m_dom = dominant_freq_distance(thetaA, thetaB)
            m_ent = spectral_entropy_dist(thetaA, thetaB)
            m_wav = wavelet_distance(thetaA, thetaB)
            m_trd = trend_distance(thetaA, thetaB)
            m_sea = seasonality_distance(thetaA, thetaB)
            m_sta = stationarity_distance(thetaA, thetaB)
            m_vol = volatility_distance(thetaA, thetaB)

            results["DTW"].append(m_dtw)
            results["FFT"].append(m_fft)
            results["DominantFreq"].append(m_dom)
            results["SpectralEntropy"].append(m_ent)
            results["Wavelet"].append(m_wav)
            results["Trend"].append(m_trd)
            results["Seasonality"].append(m_sea)
            results["Stationarity"].append(m_sta)
            results["Volatility"].append(m_vol)

            print(f"Seg {i} vs {j}:  DTW={m_dtw:.3f}, FFT={m_fft:.3f}, Dom={m_dom:.3f}, "
                  f"Ent={m_ent:.3f}, Wav={m_wav:.3f}, Trend={m_trd:.3f}, "
                  f"Season={m_sea:.3f}, Stat={m_sta:.3f}, Vol={m_vol:.3f}")

    print("\n\n================ WORD-LEVEL AVERAGE RESULTS ================\n")
    for m in metric_names:
        print(f"{m:15s} : {np.mean(results[m]):.4f}")

    print("\n===========================================================\n")

    # Representative overlay plot
    plot_overlay(wordA[0], wordB[0], nameA, nameB)

    return results



# ================================================================
# OVERLAY PLOT
# ================================================================
def plot_overlay(segA, segB, nameA, nameB):
    plt.figure(figsize=(10,4))
    plt.plot(segA[:,0], label=f"{nameA} – θ")
    plt.plot(segB[:,0], label=f"{nameB} – θ")
    plt.title(f"Overlay of Representative Segments")
    plt.xlabel("Frame Index")
    plt.ylabel("Theta")
    plt.legend()
    plt.tight_layout()
    plt.show()



# ================================================================
# RUN EXAMPLE
# ================================================================
word1 = load_segments("d10bahaar.npz")
word2 = load_segments("d10dard.npz")
word3 = load_segments("d10bhookh.npz")

compare_words(word1, word2, nameA="bahaar", nameB="dard")
compare_words(word1, word3, nameA="bahaar", nameB="bhookh")
compare_words(word2, word3, nameA="dard", nameB="bhookh")
