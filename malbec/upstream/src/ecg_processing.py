import glob
import os
import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import iirnotch, filtfilt, signal
from tqdm import tqdm


ecglookup = [
    "ECG_I",
    "ECG_II",
    "ECG_III",
    "ECG_aVR",
    "ECG_aVL",
    "ECG_aVF",
    "ECG_V1",
    "ECG_V2",
    "ECG_V3",
    "ECG_V4",
    "ECG_V5",
    "ECG_V6",
]


def butter_highpass_filter(data, fs, cutoff=0.05, order=4):
    """
    Applies a Butterworth highpass filter to the input data.

    This function designs a digital Butterworth highpass filter, which attenuates
    frequencies below the specified cutoff frequency while allowing higher frequencies
    to pass through. The filtered data is returned.

    Parameters:
    -----------
    data : array-like
        The input signal to be filtered.
    fs : float
        The sampling frequency of the input signal.
    cutoff : float, optional, default=0.05 Hz
        The cutoff frequency of the highpass filter.
    order : int, optional, default=4
        The order of the Butterworth filter. A higher order results in a sharper
        transition between the passband and the stopband.

    Returns:
    --------
    y : array-like
        The filtered signal with low frequencies attenuated.
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype="high", analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def notch_filter(data, fs, freq=50, harmonics=5, quality_factor=30):
    """
    Applies a series of notch filters to remove 50Hz and its harmonics.

    Parameters:
    -----------
    data : array-like
        The input signal to be filtered.
    fs : float
        The sampling frequency.
    freq : float, optional (default=50)
        The base frequency to be removed (e.g., 50 Hz for power line noise).
    harmonics : int, optional (default=5)
        Number of harmonics to remove (e.g., 50 Hz, 100 Hz, 150 Hz, etc.).
    quality_factor : float, optional (default=30)
        The quality factor for the notch filter. Higher values result in a narrower notch.

    Returns:
    --------
    filtered_data : array-like
        The filtered signal.
    """
    filtered_data = data
    for k in range(1, harmonics + 1):  # Apply to harmonics
        notch_freq = k * freq
        w0 = notch_freq / (fs / 2)  # Normalize frequency to Nyquist
        b, a = iirnotch(w0, quality_factor)
        filtered_data = filtfilt(b, a, filtered_data)
    return filtered_data


def downsample(y, fs=500, samples=4096):
    """
    Downsamples the input signal to a specified number of samples using linear interpolation.

    This function reduces the number of data points in the input signal `y` by
    interpolating the signal to a specified number of samples. The resulting
    downsampled signal maintains the overall shape and characteristics of the original
    signal but with fewer data points.

    Parameters:
    -----------
    y : array-like
        The input signal to be downsampled.
    fs : int, optional, default=500
        The sampling frequency of the input signal.
    samples : int, optional, default=4096
        The number of samples to downsample the signal to.

    Returns:
    --------
    down : array-like
        The downsampled signal with the specified number of samples.
    """
    downsample = np.arange(0, fs*10, (fs*10)/samples)
    interp_func = interp1d(
        np.arange(0, len(y)), y, kind="linear", fill_value="extrapolate"
    )
    down = interp_func(downsample)
    return down


def read_sch_h5(file_name):
    """
    Read, filter, and downsample ECG data from a SCH file.
    
    Parameters:
    file_name (str): The path to the SCH ECG file.

    Returns:
    numpy.ndarray: The ECG data.
    """
    if True:
        with h5py.File(file_name, "r") as f:
            fs = f["rhythms"].attrs["fs"]
            resExp = f["rhythms"].attrs["resolutionExp"]
            resf = f["rhythms"].attrs["resolutionFactor"]
            unit = f["rhythms"].attrs["unit"]

            if (
                resf == 1
                and resExp == -6
                and unit == b"Volt"
                and f["rhythms"][list(f["rhythms"].keys())[1]].shape[0] >= fs * 10
                and fs>=500
                and all(item in list(f["rhythms"].keys()) for item in ecglookup)
            ):
                data = np.zeros((1, len(ecglookup), 4096))
                for i in range(len(ecglookup)):
                    lead_ =f["rhythms"][ecglookup[i]][:].transpose()[
                                0, range(0, int(fs * 10))]
                    lead_ = notch_filter(
                           lead_, fs,50.0,int(fs/2/50-1)
                        )
                    lead_ = notch_filter(
                           lead_, fs,60.0,int(fs/2/60-1)
                        )
                    
                    lead_ = butter_highpass_filter(lead_,fs)

                    # Downsample to 4096 samples
                    data[0, i, :] = (
                        downsample(
                            lead_,
                            fs=fs,samples=4096,
                        )
                    )
                return data
            else:
                print(file_name + ": Format not supported")
    return None


def convert_sch_h5(path_list, outpath):
    """
    Convert SCH ECG files to a single HDF5 file.

    Parameters:
    path_list (list): A list of file paths to the SCH ECG files.
    outpath (str): Path and filename for the output HDF5 file.

    Returns:
    None
    """
    # Gather all files
    files = [file for path in path_list for file in glob.glob(path)]
    
    ecg = np.zeros([len(files), 12, 4096])
    hash = [None] * len(files)
    j = 0
    pbar = tqdm(files, desc="Converting ECG files", unit="file")
    for v in pbar:
        data = read_sch_h5(v)
        if(data is not None and np.isnan(data).any() == False):
            ecg[j, :, :] = data
            hash[j] = os.path.splitext(os.path.basename(v))[0]
            j += 1
        pbar.set_postfix({"Valid Files": j})

    with h5py.File(outpath, "w") as hf:
        hf.create_dataset("data", data=ecg[0:j,:,:])
        hf.create_dataset("hash", data=np.array(hash[0:j]).astype("S"))

    return None