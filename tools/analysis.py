import numpy as np 
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
import numpy as np 
import seaborn as sns 

def plot_frequency_spectrum(raw_eeg, channel_name):
    data, times = raw_eeg[channel_name, :]
    fft_data = np.fft.rfft(data.flatten())
    freqs = np.fft.rfftfreq(len(data.flatten()), d=1/raw_eeg.info['sfreq'])
    psd = np.abs(fft_data) ** 2
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd)
    plt.xlim(0, 50)
    plt.title(f'Frequency Spectrum for Channel: {channel_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (uV^2/Hz)')
    plt.grid()
    plt.show()
    
def rieamnn_cov(epochs):
    ch_epochs = epochs.copy().pick(['Fp1', 'Fpz', 'Fp2'])
    #epochs.plot_image(picks=['Fp1', 'Fpz', 'Fp2'])
    data = ch_epochs.get_data(copy=False)
    cov_matrices = Covariances('oas').fit_transform(data)
    mean_cov = mean_riemann(cov_matrices)
    distances = np.array([distance_riemann(cov, mean_cov) for cov in cov_matrices])
    threshold = np.mean(distances) + 2*np.std(distances)
    blinks = distances > threshold

    # Вывод результатов
    print(f"Среднее расстояние: {np.mean(distances):.4f}")
    print(f"Порог аномалии: {threshold:.4f}")
    print(f"Количество обнаруженных морганий: {np.sum(blinks)}")

    # Построение графиков удаленных эпох
    epochs_to_drop_indices = np.where(blinks)[0]
    #deleted_epochs = epochs[epochs_to_drop_indices]
    #deleted_epochs.plot(picks=['Fp1', 'Fpz', 'Fp2'])
    epochs = epochs.drop(epochs_to_drop_indices)
    return epochs

def cov_analysis(raw, coef=1, plot=False):
    cov = np.cov(raw.get_data())
    mean_cov = np.mean(cov)
    std_cov = np.std(cov)
    threshold = mean_cov + coef*std_cov
    bad_channels = np.where(np.mean(cov, axis=1) > threshold)[0]
    channel_names = raw.info['ch_names']
    bad_channel_names = [channel_names[idx] for idx in bad_channels]
    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(cov, cmap='viridis', origin='upper')
        plt.colorbar(label='Covariance')
        plt.title('Covariance Matrix')
        plt.xlabel('Channels')
        plt.ylabel('Channels')
        plt.show()
    return bad_channel_names, cov

def rms_calculate(epochs):
    evoked = epochs.average()
    evoked_data = evoked.data
    epochs_data = epochs.get_data()
    noise_epochs = epochs_data - evoked_data
    rms_signal = np.sqrt(np.mean(evoked_data**2, axis=1)) # calculate etalon-signal, returned channel noise (n_channels)
    rms_noise = np.sqrt(np.mean(noise_epochs**2, axis=2)) # calculate timeline-noise, returned channels noise, (n_epochs, n_channels)
    return rms_signal, rms_noise

def snr_calculate(rms_signal, rms_noise, mode='ch'):
    snr_matrix = np.zeros(rms_noise.shape[:2])
    for ep in range(snr_matrix.shape[0]):
        for ch in range(snr_matrix.shape[1]):
            if mode=='mean':
                snr_matrix[ep][ch] = 20 * np.log10(np.mean(rms_signal, axis=0) / rms_noise[ep])[ch]
            if mode =='ch':
                snr_matrix[ep][ch] = 20 * np.log10(rms_signal[ch] / rms_noise[ep])[ch]
    return snr_matrix

def snr_plot(snr_matrix, ep_ind, ch_ind, ch_names, save=None):
    plt.figure(figsize=(8, 6))
    xticklabels=np.array(ep_ind)[::len(ep_ind)//10]
    yticklabels=np.array(ch_names)[ch_ind]
    sns.heatmap(snr_matrix[np.ix_(ep_ind,ch_ind)].T, cmap='magma',  xticklabels=xticklabels, yticklabels=yticklabels, vmin=-35, vmax=0)
    plt.xticks(ticks=xticklabels, labels=np.array(ep_ind)[::len(ep_ind)//10])
    plt.title('SNR Heatmap')
    plt.xlabel('Epochs')
    plt.ylabel('Channels')
    plt.show()
    if save:
        plt.savefig(save)