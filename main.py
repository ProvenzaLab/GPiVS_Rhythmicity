import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy

def loadmat(filename) -> dict:
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    from scipy.io import loadmat as sio_loadmat

    data = sio_loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    from scipy.io.matlab import mat_struct

    for key in dict:
        if isinstance(dict[key], mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj) -> dict:
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    from scipy.io.matlab import mat_struct

    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def main():
    PATH_DATA = '/Users/Timon/Documents/Houston/Circadian Rhythmicity Project GPiVCVS/Vigi/GPi_OCD/Matlab/BrainSense.mat'
    g_ = []
    v_ = []
    t_ = []
    for i in range(3):
        g_.append(_todict(loadmat(PATH_DATA)["G"][i])["sA"])
        v_.append(_todict(loadmat(PATH_DATA)["V"][i])["sA"])
        t_.append(_todict(loadmat(PATH_DATA)["V"][i])["tA"])
    
    v_.append(_todict(loadmat(PATH_DATA)["V"][3])["sA"])
    g_.append(_todict(loadmat(PATH_DATA)["G"][4])["sA"])
    t_.append(_todict(loadmat(PATH_DATA)["V"][3])["tA"])

    v_.append(_todict(loadmat(PATH_DATA)["V"][4])["sA"])
    g_.append(_todict(loadmat(PATH_DATA)["G"][5])["sA"])
    t_.append(_todict(loadmat(PATH_DATA)["V"][4])["tA"])

    v_.append(_todict(loadmat(PATH_DATA)["V"][5])["sA"])
    g_.append(_todict(loadmat(PATH_DATA)["G"][6])["sA"])
    t_.append(_todict(loadmat(PATH_DATA)["V"][5])["tA"])

    v_.append(_todict(loadmat(PATH_DATA)["V"][6])["sA"])
    g_.append(_todict(loadmat(PATH_DATA)["G"][9])["sA"])
    t_.append(_todict(loadmat(PATH_DATA)["V"][6])["tA"])

    g_ = np.concatenate(g_)
    v_ = np.concatenate(v_)
    t_ = np.concatenate(t_)

    plt.plot(t_, g_, label='G_t')  
    plt.plot(t_, v_, label='V_t')
    plt.xlabel('Time (s)')


    from mne import create_info
    from mne.io import RawArray
    import mne_connectivity

    sfreq = 250  # Sampling frequency
    ch_names = ['G_t', 'V_t']
    ch_types = ['dbs', 'dbs']
    info = create_info(ch_names, sfreq, ch_types)
    # shuffle g_
    #np.random.shuffle(g_)
    raw = RawArray(np.array([g_, v_]), info)
    epochs = mne.make_fixed_length_epochs(raw, duration=10, preload=True)
    conn = mne_connectivity.spectral_connectivity_time(
        epochs.get_data(),
        freqs=np.arange(2, 50, 1),
        method='plv',  # coh
        indices=(np.array([0, 1]), np.array([1, 0])),
        sfreq=250,
        n_cycles=7,
        n_jobs=1
    )
    conn_data = conn.get_data()

    # shuffle g_
    g_s = g_.copy()
    np.random.shuffle(g_s)
    raw_s = RawArray(np.array([g_s, v_]), info)
    epochs_s = mne.make_fixed_length_epochs(raw_s, duration=10, preload=True)
    conn_shuffled = mne_connectivity.spectral_connectivity_time(
        epochs_s.get_data(),
        freqs=np.arange(2, 50, 1),
        method='plv',
        indices=(np.array([0, 1]), np.array([1, 0])),
        sfreq=250,
        n_cycles=5,
        n_jobs=1
    )
    conn_data_s = conn_shuffled.get_data()

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    f_, Pxx = scipy.signal.welch(g_, fs=250, nperseg=2*250, axis=0)
    # limit the frequency to 50 Hz
    f = f_[f_ < 50]
    Pxx = Pxx[f_ < 50]
    plt.plot(f, np.log10(Pxx), label='GPi')
    f, Pxx = scipy.signal.welch(v_, fs=250, nperseg=2*250, axis=0)
    f = f_[f_ < 50]
    Pxx = Pxx[f_ < 50]
    plt.plot(f, np.log10(Pxx), label='VS')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title('Welch Power Spectral Density')
    plt.legend()
    plt.xlim(0, 50)
    
    plt.subplot(2, 2, 2)
    spectrum = epochs.compute_psd(method="multitaper", fmin=2, fmax=60, n_jobs=1)
    idx_50_hz = np.where(spectrum.freqs > 50)[0][0]
    plt.imshow(np.log10(spectrum.get_data()[:, 0, :idx_50_hz]).T, aspect='auto')
    # flip the y-axis
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Time Frequency PSD GPi")
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(0, len(spectrum.freqs[:idx_50_hz]), 20), np.round(spectrum.freqs[:idx_50_hz][::20], 2).astype(int))
    cbar = plt.colorbar()
    cbar.set_label("Power (dB)")
    
    plt.subplot(2, 2, 4)
    power_sum = conn_data[:, 0, :].sum(axis=1)
    plt.imshow((conn_data[:, 0, :] / np.expand_dims(power_sum, axis=1)).T, aspect='auto')
    plt.gca().invert_yaxis()
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Time-resolved coherence GPi - VS")
    cbar = plt.colorbar()
    cbar.set_label("Coherence")

    plt.subplot(2, 2, 3)
    PLV = True
    if PLV:
        plt.plot(np.nanmean(conn_data[:, 0, :], axis=0))
    else:
        coh_normed = (conn_data[:, 0, :] / np.expand_dims(power_sum, axis=1)).mean(axis=0)
        coh_std = (conn_data[:, 0, :] / np.expand_dims(power_sum, axis=1)).std(axis=0)
        power_sum_s = conn_data_s[:, 0, :].sum(axis=1)
        coh_normed_s = (conn_data_s[:, 0, :] / np.expand_dims(power_sum, axis=1)).mean(axis=0)
        coh_std_s = (conn_data_s[:, 0, :] / np.expand_dims(power_sum, axis=1)).std(axis=0)

        plt.plot(conn.freqs, coh_normed_s, color="gray", label="Shuffled")
        plt.fill_between(conn.freqs, coh_normed_s - coh_std_s, coh_normed_s + coh_std_s, alpha=0.5, color="gray")
        plt.plot(conn.freqs, coh_normed, color="black", label="Original")
        plt.fill_between(conn.freqs, coh_normed - coh_std, coh_normed + coh_std, alpha=0.5, color="black")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.legend()
    plt.title("Mean coherence GPi - VS")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
