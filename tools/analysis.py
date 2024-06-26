import numpy as np 
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
import numpy as np 

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
