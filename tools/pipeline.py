import numpy as np 
import mne 
from mne.preprocessing import ICA, compute_proj_ecg, compute_proj_eog
from .quality_check import detect_noise_channel_neigbours, event_check, bridging_test
from mne_icalabel import label_components
from autoreject import AutoReject 
from .montages import create_custom_montage
from .savers import components_save
import time
import matplotlib.pyplot as plt 
from jinja2 import Environment, FileSystemLoader
import re
import os 
from scipy.fft import fft
from tools.montages import read_elc, align_head
from autoreject import Ransac 
from tools.analysis import rms_calculate, snr_calculate, snr_plot
from matplotlib.gridspec import GridSpec

def erps(raw_filtered):
    modes = ['without', 'central', 'space', 'all']
    for mode in modes:
        if mode =='without':
            events_congr = ['Stimulus/s145']
            events_not_congr = ['Stimulus/s147']
        if mode == 'all':
            events_congr = ['Stimulus/s143', 'Stimulus/s145', 'Stimulus/s149', 'Stimulus/s151']
            events_not_congr = ['Stimulus/s141', 'Stimulus/s147', 'Stimulus/s153', 'Stimulus/s155']
        if mode == 'central':
            events_congr = ['Stimulus/s143']
            events_not_congr = ['Stimulus/s141']
        if mode == 'space':
            events_congr = ['Stimulus/s149', 'Stimulus/s151']
            events_not_congr = ['Stimulus/s153', 'Stimulus/s155']

        def erps_events(raw_filtered, events_congr, events_not_congr):
            def make_events(target_stimuli, raw, flag):
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                desired_stimulus = 'Stimulus/s200'
                target_stimuli_codes = dict(list(zip(event_id.values(), event_id.keys())))
                filtered_events = []
                for i in range(len(events) - 1):
                    current_event = events[i]
                    next_event = events[i + 1]
                    if target_stimuli_codes[current_event[2]] in target_stimuli:
                        if target_stimuli_codes[next_event[2]] == desired_stimulus:
                            filtered_events.append(current_event)
                filtered_event_id = {stimulus: event_id[stimulus] for stimulus in target_stimuli if stimulus in event_id}
                filtered_events = np.array(filtered_events)
                epochs_noise = mne.Epochs(raw, filtered_events, event_id=filtered_event_id, tmin=-0.2, tmax=1.0, preload=True, 
                                    baseline=(None, 0), event_repeated='merge', verbose=False)
                
                rms_signal, rms_noise = rms_calculate(epochs_noise)
                snr_matrix = snr_calculate(rms_signal, rms_noise, 'mean')
                n_epochs = epochs_noise.get_data().shape[0]
                ch_names = epochs_noise.ch_names
                snr_plot(snr_matrix, list(range(n_epochs)), list(range(len(ch_names))), ch_names, save=f'SNR-{mode}-{flag}')
                
                threshold_ep = -25
                threshold_elec = -28

                # Вычисление среднего значения по каналам для каждой эпохи
                mean_snr_per_epoch = np.mean(snr_matrix, axis=1)
                # Нахождение индексов эпох, где среднее значение по каналам меньше порога
                epochs_below_threshold = np.where(mean_snr_per_epoch < threshold_ep)[0]
                print("Индексы эпох, где среднее значение по каналам меньше порога:", epochs_below_threshold)
                # Для каждой интересующей нас эпохи, находим имена каналов, которые превысили threshold1
                for epoch in range(n_epochs):
                    channels_above_threshold1 = np.where(snr_matrix[epoch] < threshold_elec)[0]
                    channel_names_above_threshold1 = [ch_names[ch] for ch in channels_above_threshold1]
                    print(f"Эпоха {epoch}: Каналы, превысившие порог {threshold_elec}: {channel_names_above_threshold1}")
                    
                reject_criteria = dict(eeg=130e-6)
                epochs_cleaned = epochs_noise.copy().drop_bad(reject=reject_criteria, verbose=False)
                drop_log = epochs_cleaned.drop_log
                rejected_epochs = [i for i, log in enumerate(drop_log) if log]
                print(f'Удаленные эпохи {flag}: {rejected_epochs}')
                return epochs_cleaned, rejected_epochs

            epochs_c, rej_c = make_events(events_congr, raw_filtered, flag='c')
            epochs_n, rej_n = make_events(events_not_congr, raw_filtered, flag='n')
            
            #ar = AutoReject(thresh_method='random_search', random_state=42)

            #ar.fit(epochs_c)
            #epochs_c = ar.transform(epochs_c)
            #ar.fit(epochs_n)
            #epochs_n = ar.transform(epochs_n)
            erp = {'P1':['P3', 'Pz', 'P4', 'POz', 'P1', 'P2'],
                'N1':['P3', 'Pz', 'P4', 'O1', 'Oz', 'O2'],
                'CNV':['FCz', 'Cz'],
                'N2':['P3', 'Pz', 'P4', 'POz', 'P1', 'P2'],
                'P3':['P3', 'Pz', 'P4', 'POz', 'P1', 'P2'],}
            electrodes_to_plot = erp['P3']
            n_epochs_c = len(epochs_c)
            n_epochs_n = len(epochs_n)
            num_electrodes = len(electrodes_to_plot)
            baseline = 0.2
            sf = raw_filtered.info['sfreq'] 
            n_fft= int(sf)

            fig_erp = plt.figure(figsize=(16, 12))  # Изменим размер фигуры для сетки 4:2
            gs_erp = GridSpec(4, 2, figure=fig_erp, hspace=0.7)
            fig_erp.suptitle(f'ERP C:{n_epochs_c}/{n_epochs_c+len(rej_c)}, N:{n_epochs_n}/{n_epochs_n+len(rej_n)}')

            for i, electrode in enumerate(electrodes_to_plot):
                row, col = divmod(i, 2)
                ax_erp = fig_erp.add_subplot(gs_erp[row, col])
                data_c = epochs_c.get_data(picks=electrode)
                data_n = epochs_n.get_data(picks=electrode)
                mean_data_c = np.mean(data_c, axis=0).squeeze()
                mean_data_n = np.mean(data_n, axis=0).squeeze()
                std_c = np.std(data_c, axis=0).squeeze()
                std_n = np.std(data_n, axis=0).squeeze()
                print(mean_data_c.shape)
                M_c = np.mean(mean_data_c, axis=0)
                SD_c = np.std(mean_data_c, axis=0)
                M_n = np.mean(mean_data_n, axis=0)
                SD_n = np.std(mean_data_n, axis=0)
                t = np.arange(-baseline, len(mean_data_c) / sf - baseline, 1 / sf)
                # График ERP
                ax_erp.plot(t, mean_data_c, label='con', color='blue')
                ax_erp.plot(t, mean_data_n, label='n-con', color='red')
                ax_erp.fill_between(t, mean_data_c - std_c, mean_data_c + std_c, color='blue', alpha=0.3)
                ax_erp.fill_between(t, mean_data_n - std_n, mean_data_n + std_n, color='red', alpha=0.3)
                ax_erp.set_ylim([-15 * 1e-6, 15 * 1e-6])
                ax_erp.set_title(f'ERP {electrode}, C: {M_c*1e6:.2f}$\pm${SD_c*1e6:.2f} $\mu$V, N: {M_n*1e6:.2f}$\pm${SD_n*1e6:.2f} $\mu$V', fontsize=10)
                ax_erp.set_xlabel('Время (с)', fontsize=8)
                ax_erp.set_ylabel('Амплитуда', fontsize=8)
                ax_erp.axhline(0, color='black', linewidth=1)
                ax_erp.axvline(0, color='black', linewidth=1)
                ax_erp.grid(True)
                ax_erp.legend(fontsize=8)
                ax_erp.set_box_aspect(0.2)  # Убедимся, что графики имеют одинаковое соотношение сторон

            # Среднее по электродам ERP
            mean_data_c_all = np.mean([epochs_c.average(picks=electrode, method='mean').data for electrode in erp['P3']], axis=0).squeeze()
            mean_data_n_all = np.mean([epochs_n.average(picks=electrode, method='mean').data for electrode in erp['P3']], axis=0).squeeze()
            SD_c_all = np.std([epochs_c.average(picks=electrode, method='mean').data for electrode in erp['P3']], axis=0).squeeze()
            SD_n_all = np.std([epochs_n.average(picks=electrode, method='mean').data for electrode in erp['P3']], axis=0).squeeze()
            M_c = np.mean(mean_data_c_all, axis=0)
            SD_c = np.std(mean_data_c_all, axis=0)
            M_n = np.mean(mean_data_n_all, axis=0)
            SD_n = np.std(mean_data_n_all, axis=0)
            t = np.arange(-baseline, len(mean_data_c_all) / sf - baseline, 1 / sf)

            # Добавление графика среднего ERP с теневым контуром
            ax_erp_avg = fig_erp.add_subplot(gs_erp[3, :])
            ax_erp_avg.plot(t, mean_data_c_all, label='con', color='blue', linestyle='--')
            ax_erp_avg.plot(t, mean_data_n_all, label='n-con', color='red', linestyle='--')

            ax_erp_avg.fill_between(t, mean_data_c_all - SD_c_all,
                                    mean_data_c_all + SD_c_all, color='blue', alpha=0.3)
            ax_erp_avg.fill_between(t, mean_data_n_all - SD_n_all,
                                    mean_data_n_all + SD_n_all, color='red', alpha=0.3)

            ax_erp_avg.set_ylim([-15 * 1e-6, 15 * 1e-6])
            ax_erp_avg.set_title(f'ERP Среднее по электродам, C: {M_c*1e6:.2f}$\pm${SD_c*1e6:.2f} $\mu$V, N: {M_n*1e6:.2f}$\pm${SD_n*1e6:.2f} $\mu$V', fontsize=10)
            ax_erp_avg.set_xlabel('Время (с)', fontsize=8)
            ax_erp_avg.set_ylabel('Амплитуда', fontsize=8)
            ax_erp_avg.axhline(0, color='black', linewidth=1)
            ax_erp_avg.axvline(0, color='black', linewidth=1)
            ax_erp_avg.grid(True)
            ax_erp_avg.legend(fontsize=8)
            ax_erp_avg.set_box_aspect(0.2)

            plt.tight_layout()
            plt.show()
            fig_erp.savefig(f'ERP_P3_{mode}.png')
            # Создаем второе окно для PSD
            fig_psd = plt.figure(figsize=(16, 12))  # Изменим размер фигуры для сетки 4:2
            gs_psd = GridSpec(4, 2, figure=fig_psd, hspace=0.7)
            fig_psd.suptitle('PSD')
            for i, electrode in enumerate(electrodes_to_plot):
                row, col = divmod(i, 2)
                ax_psd = fig_psd.add_subplot(gs_psd[row, col])

                mean_data_c = np.mean(epochs_c.average(picks=electrode, method='mean').data, axis=0)
                mean_data_n = np.mean(epochs_n.average(picks=electrode, method='mean').data, axis=0)
                
                # Расчет и график спектра
                psd_c, freqs_c = mne.time_frequency.psd_array_welch(mean_data_c, sfreq=sf, fmin=0, fmax=50, n_fft=n_fft)
                psd_n, freqs_n = mne.time_frequency.psd_array_welch(mean_data_n, sfreq=sf, fmin=0, fmax=50, n_fft=n_fft)

                ax_psd.plot(freqs_c, 10 * np.log10(psd_c), label='con', color='blue')
                ax_psd.plot(freqs_n, 10 * np.log10(psd_n), label='n-con', color='red')
                ax_psd.set_title(f'PSD {electrode}', fontsize=10)
                ax_psd.set_xlabel('Частота (Гц)', fontsize=8)
                ax_psd.set_ylabel('Мощность (дБ)', fontsize=8)
                ax_psd.axhline(-100, color='black', linewidth=1)
                ax_psd.axvline(0, color='black', linewidth=1)
                ax_psd.grid(True)
                ax_psd.legend(fontsize=8)
                ax_psd.set_box_aspect(0.2)  # Убедимся, что графики имеют одинаковое соотношение сторон

            # Среднее по электродам PSD
            psd_c_all, freqs_c_all = mne.time_frequency.psd_array_welch(mean_data_c_all, sfreq=sf, fmin=0, fmax=50, n_fft=n_fft)
            psd_n_all, freqs_n_all = mne.time_frequency.psd_array_welch(mean_data_n_all, sfreq=sf, fmin=0, fmax=50, n_fft=n_fft)

            ax_psd_avg = fig_psd.add_subplot(gs_psd[3, :])
            ax_psd_avg.plot(freqs_c_all, 10 * np.log10(psd_c_all), label='con', color='blue', linestyle='--')
            ax_psd_avg.plot(freqs_n_all, 10 * np.log10(psd_n_all), label='n-con', color='red', linestyle='--')
            ax_psd_avg.set_title(f'PSD Среднее по электродам', fontsize=10)
            ax_psd_avg.set_xlabel('Частота (Гц)', fontsize=8)
            ax_psd_avg.set_ylabel('Мощность (дБ)', fontsize=8)
            ax_psd_avg.axhline(-100, color='black', linewidth=1)
            ax_psd_avg.axvline(0, color='black', linewidth=1)
            ax_psd_avg.grid(True)
            ax_psd_avg.legend(fontsize=8)
            ax_psd_avg.set_box_aspect(0.1)
            
            plt.tight_layout()
            plt.show()
            fig_psd.savefig(f'PSD_P3_{mode}.png')

        erps_events(raw_filtered, events_congr, events_not_congr)

class Transform:
    def __call__(self, raw):
        raw = raw.copy()
        return self.forward(raw)
    def forward(self, raw):
        raise NotImplementedError("Each transform must implement the forward method.")
    
class Sequence:
    def __init__(self, *transforms):
        self.transforms = transforms
    
    def __call__(self, raw):
        for transform in self.transforms:
            raw = transform(raw)
        return raw
    
class ExcludeECGEOG(Transform):
    def __init__(self, ecg_ch, eog_ch):
        self.ecg_ch = ecg_ch
        self.eog_ch = eog_ch
    def forward(self, raw):
        return exclude_ecgeog(raw, self.ecg_ch, self.eog_ch)
def exclude_ecgeog(raw, ecg_ch, eog_ch):
    ecg_projs, ecg_events = compute_proj_ecg(raw, ch_name=ecg_ch, average=True, verbose=False)
    eog_projs, eog_events = compute_proj_eog(raw, ch_name=eog_ch, average=True, verbose=False)
    raw = raw.add_proj(ecg_projs)
    raw = raw.add_proj(eog_projs)
    raw = raw.apply_proj()
    raw.info.normalize_proj()
    raw = raw.drop_channels([ecg_ch, eog_ch])
    return raw

class Cropping(Transform):
    def __init__(self, stimulus=None):
        if stimulus:
            self.stimulus = stimulus 
        else:
            self.stimulus = ['Stimulus/s50', 'Stimulus/s52', 'Stimulus/s54', 'Stimulus/s56', 'Stimulus/s58',
                            'Stimulus/s60', 'Stimulus/s62', 'Stimulus/s64', 'Stimulus/s66', 'Stimulus/s68',
                            'Stimulus/s70', 'Stimulus/s72', 'Stimulus/s74', 'Stimulus/s76', 'Stimulus/s78', 
                            'Stimulus/s80']
    def forward(self, raw):
        events, event_id = mne.events_from_annotations(raw)
        stimulus_codes = [event_id[stim] for stim in self.stimulus if stim in event_id]
        for event in events:
            if event[2] in stimulus_codes:
                first_stimulus_time = event[0] / raw.info['sfreq']
                break
        for event in events[::-1]:
            if event[2] in stimulus_codes:
                last_stimulus_time = event[0] / raw.info['sfreq']
                break
        tmin = max(0, first_stimulus_time-1.0)
        tmax = min(raw.times[-1], last_stimulus_time + 1.0)
        raw = raw.crop(tmin=tmin, tmax=tmax)
        return raw

class FilterBandpass(Transform):
    def __init__(self, low_freq, high_freq, notch_freq=False, method='firwin'):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.notch_freq = notch_freq
        self.method = method
    def forward(self, raw):
        raw = raw.filter(l_freq=self.low_freq, h_freq=self.high_freq, fir_design=self.method, verbose=False, pad='reflect_limited', phase='zero-double')
        if self.notch_freq:
            raw = raw.notch_filter(freqs=self.notch_freq, fir_design=self.method, verbose=False)
        return raw

class SetMontage(Transform):
    def __init__(self, montage, elc_file=None, mode='Cz', vis=False, threshold=0.08, interpolate=True):
        self.montage = montage
        self.elc_file = elc_file
        self.vis = vis
        self.mode = mode
        self.threshold = threshold
        self.interpolate = interpolate
    def forward(self, raw):
        if self.montage=='waveguard64':
            montage = create_custom_montage(self.montage)
        elif self.montage == 'personal':
            ch_dict, nasion, lpa, rpa, hsp = read_elc(self.elc_file)
            if self.interpolate:
                ch_dict, nasion, lpa, rpa, hsp = align_head(ch_dict, nasion, np.array(lpa), np.array(rpa), np.array(hsp), standard='waveguard64', 
                                                        mode=self.mode, threshold=self.threshold)
            montage = mne.channels.make_dig_montage(ch_pos=ch_dict, nasion=nasion, lpa=lpa, rpa=rpa, hsp=hsp, coord_frame='head')
        else:
            montage = mne.channels.make_standard_montage(self.montage)
        channels_to_drop = list((set(raw.info['ch_names'])) - (set(raw.info['ch_names']) & set(montage.ch_names)))
        print(f'CHANNELS DROPPED: {channels_to_drop}')
        raw.drop_channels(channels_to_drop)
        raw.set_montage(montage, verbose=False)
        if self.vis:
            fig = montage.plot(show_names=True, kind='3d')
            for ax in fig.get_axes():
                if ax.name == '3d':
                    ax.set_xlim([-0.1, 0.1])
                    ax.set_ylim([-0.1, 0.1])
                    ax.set_zlim([-0.1, 0.1])
        return raw

class BridgeInterpolate(Transform):
    def __init__(self, bridged_idx):
        self.bridged_idx = bridged_idx
    def forward(self, raw):
        raw = mne.preprocessing.interpolate_bridged_electrodes(raw.copy(), self.bridged_idx, bad_limit=4)
        return raw

class Interpolate(Transform):
    def __init__(self, bad_channels=None, method='ransac', bad_limit=4):
        self.bad_channels = bad_channels
        self.bad_limit = bad_limit
        self.method = method
    def forward(self, raw):
        if self.bad_channels is None:
            print(f'<Noise Detector>:\nNoised channels will be found automatically')
            if self.method=='ransac':
                ransac = Ransac(verbose=False)
                events, event_id = mne.events_from_annotations(raw)
                epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=1.0, preload=True, baseline=(None, 0), event_repeated='merge')
                _ = ransac.fit_transform(epochs)
                self.bad_channels = ransac.bad_chs_
            if self.method=='neighbour':
                noisy_channels, score = detect_noise_channel_neigbours(raw, level=0.1)
                self.bad_channels = [x[0] for x in sorted(zip(noisy_channels, score), key=lambda x: x[1], reverse=True)]
        raw.info['bads'] += self.bad_channels[:self.bad_limit]
        raw.interpolate_bads(reset_bads=True)
        print(f'{self.bad_channels} have been interpolated')
        return raw
    
class Rereference(Transform):
    def __init__(self, method='average'):
        self.method = method
    def forward(self, raw):
        raw = raw.set_eeg_reference(self.method, verbose=False)
        return raw 

class Resample(Transform):
    def __init__(self, sfreq):
        self.sfreq = sfreq
    def forward(self, raw):
        raw = raw.resample(sfreq=self.sfreq, npad="auto")
        return raw
    
class AutoICA(Transform):
    def __init__(self, n_components=20, threshold=0, vis=False, saving_dir=None, method='infomax', mode='auto', max_iter='auto', apply=True):
        self.n_components = n_components
        self.method = method
        self.max_iter=max_iter
        self.threshold = threshold
        self.saving_dir = saving_dir
        self.vis = vis
        self.mode = mode
        self.apply = apply
        self.ica = ICA(n_components=self.n_components, method=self.method, max_iter=max_iter, 
                            random_state=97, fit_params=dict(extended=True))
    def forward(self, raw):
        self.ica.fit(raw)
        noisy_channel_names = []
        if self.mode=='hand':
            plt.ion()
            self.ica.plot_components()
            self.ica.plot_properties(raw, picks=range(0, self.n_components))
            self.ica.plot_sources(raw)
            plt.draw()
            plt.show()
            time.sleep(1)
            plt.pause(2.0)
            
            sign=True
            while sign:
                list_str = input('Enter the components separated by a space').split(' ')
                exclude_idx = [int(item) for item in list_str]
                self.ica.plot_overlay(raw, exclude=exclude_idx)
                plt.draw()
                plt.show()
                sign = bool(True if input(f'The last entered components will be excluded. Are you finished? Yes/No')=='No' else False)
                labels = None
                probas = None
                
        if self.mode=='auto':
            ica_labels = label_components(raw, self.ica, method='iclabel')
            labels = ica_labels["labels"]
            probas = ica_labels['y_pred_proba']
            exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain"] and probas[idx] > self.threshold]
            noise_channel_idx = [idx for idx, label in enumerate(labels) if label == "channel noise" and probas[idx] > self.threshold]
            noisy_map_channels = [self.ica.get_components()[:, idx] for idx in noise_channel_idx]
            for noisy_map_channel in noisy_map_channels:
                max_idx = np.argmax(noisy_map_channel)
                min_idx = np.argmin(noisy_map_channel)
                if abs(noisy_map_channel[max_idx]) > abs(noisy_map_channel[min_idx]):
                    noise_channel = raw.info['ch_names'][max_idx]
                else:
                    noise_channel = raw.info['ch_names'][min_idx]
                print('Найден зашумлённый канал', noise_channel)
                noisy_channel_names.append(noise_channel)
            exclude_info = ''.join([f'Component: {idx},\n\tclass: {labels[idx]},\n\tprobability: {probas[idx]}\n' for idx in exclude_idx])
            print(f'<Automatic markup of the ICA component>:\nArtefact components identified:\n{exclude_info}')

        components_save(raw, self.ica, exclude_idx, labels, probas, self.saving_dir, self.vis)
        if self.apply:
            raw = self.ica.apply(raw, exclude=exclude_idx)
        return raw

class ToEpoch(Transform):
    def __init__(self, tmin=-0.15, tmax=0.75, baseline=(None, 0), vis=False):
        self.tmin=tmin 
        self.tmax=tmax 
        self.baseline=baseline
        self.vis=vis
    def forward(self, raw):
        return raw2epoch(raw, self.tmin, self.tmax, self.baseline, self.vis)

def raw2epoch(raw, tmin=-0.15, tmax=0.75, baseline=(None, 0), vis=False):
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, preload=True, baseline=baseline, event_repeated='merge')
    if vis:
        epochs.plot_drop_log()
    epochs.drop_bad()
    #print({event: len(epochs[event]) for event in event_id})
    return epochs

def calculate_tfr(epochs, method, fmin=2, fmax=40, n_coef=3.5):
    sfreq=epochs.info['sfreq']
    num=int(n_coef*(fmax-fmin))
    freqs = np.logspace(*np.log10([fmin, fmax]), num=num)  
    n_cycles = (freqs*np.pi*(len(epochs.times)-1))/(5*sfreq)
    tfr, itc = epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, average=True, return_itc=True, n_jobs=7, use_fft=False, zero_mean=True)
    return tfr, itc

def create_dossier(raw, filename, saving_dir):
    event_check(raw, saving_dir=saving_dir, vis=False)
    pipeline1 = Sequence(Cropping(),
                    ExcludeECGEOG(ecg_ch='BIP1', eog_ch='EOG'),
                    FilterBandpass(low_freq=1, high_freq=100, notch_freq=50, method='firwin'),
                    SetMontage('waveguard64'),)
    raw = pipeline1(raw)
    bridged_idx = bridging_test(raw, saving_dir=saving_dir, vis=False)
    noisy_channels, score = detect_noise_channel_neigbours(raw, level=0.1)
    detector_bad_channels = [x[0] for x in sorted(zip(noisy_channels, score), key=lambda x: x[1], reverse=True)][:4]

    id_pattern = re.compile(r'INP\d{4}')
    date_pattern = re.compile(r'(\d{2}\.\d{2}\.\d{2})\.vhdr$')

    id_match = id_pattern.search(filename)
    if id_match:
        file_id = id_match.group()
    else:
        file_id = None
    date_match = date_pattern.search(filename)
    if date_match:
        file_date = date_match.group(1)
    else:
        file_date = None
    pipeline2 = Sequence(Rereference(),
                        Interpolate(bad_limit=3),
                        Resample(1024),
                        AutoICA(n_components=20, mode='auto', threshold=0.0, saving_dir=saving_dir, vis=False))
    raw = pipeline2(raw)
    file_experiment = 'ASSR' # Заглушка
    detector_bad_channels 
    file_id  
    file_date
    name_bridge_file = 'bridged_electrodes.png'
    name_ica_file = 'ica_components.png'
    name_stimulus_file = 'stimulus_counts.png'
    env = Environment(loader=FileSystemLoader('templates'))

    # Загрузка шаблона
    template = env.get_template('dossier_template.html')
    #Рендеринг шаблона с параметрами
    rendered_content = template.render(NAME=file_id, EXPERIMENT=file_experiment, DATE=file_date, 
                                       EVENT_TABLE=name_stimulus_file, BRIDGES=name_bridge_file, 
                                       COMPONENTS=name_ica_file, NOISE_CHANNELS=detector_bad_channels)
    
    # Запись сгенерированного контента в файл
    with open(os.path.join(saving_dir, 'dossier.html'), 'w') as f:
        f.write(rendered_content)


def compute_power_level(data):
    fft_vals = fft(data)
    power_spectrum = np.abs(fft_vals)**2
    noise_power = np.mean(power_spectrum)
    return noise_power

def power_detect_noise_channels(raw, level_power_theshold=1.2):
    n_channels = len(raw.info['ch_names'])
    noise_levels = np.array([compute_power_level(raw.get_data()[i]) for i in range(n_channels)])
    threshold = np.median(noise_levels) * level_power_theshold
    classification = ['Noisy' if nl > threshold else 'Clean' for nl in noise_levels]
    return sorted([(raw.info['ch_names'][i], nl, classification[i]) for i, nl in enumerate(noise_levels) if nl > threshold], key=lambda x: x[1], reverse=True)

def sliding_window_noise_plot(raw):
    def sliding_window_analysis(data, window_size, step_size):
        n_windows = (data.shape[1] - window_size) // step_size + 1
        noise_levels = np.zeros((data.shape[0], n_windows))
        for i in range(n_windows):
            start = i * step_size
            end = start + window_size
            window_data = data[:, start:end]
            for j in range(data.shape[0]):
                noise_levels[j, i] = compute_power_level(window_data[j])
        return noise_levels
    window_size = 20  # размер окна
    step_size = 5  # шаг окна
    noise_levels_windows = sliding_window_analysis(raw.get_data(), window_size, step_size)
    threshold = np.median(noise_levels_windows)*raw.info['sfreq']/2
    noise_levels_windows_new = np.where(noise_levels_windows > threshold, 1, 0)
    raw_noise = mne.io.RawArray(noise_levels_windows_new, raw.info)
    raw_noise.plot()