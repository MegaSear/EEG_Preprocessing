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
            def make_events(target_stimuli, raw):
                events, event_id = mne.events_from_annotations(raw)
                desired_stimulus = 'Stimulus/s200'
                target_stimuli_codes = dict(list(zip(event_id.values(), event_id.keys())))
                filtered_events = []
                for i in range(len(events) - 1):
                    current_event = events[i]
                    next_event = events[i + 1]
                    if target_stimuli_codes[current_event[2]] in target_stimuli:
                        if target_stimuli_codes[next_event[2]] == desired_stimulus:
                            filtered_events.append(next_event)
                filtered_events = np.array(filtered_events)
                epochs = mne.Epochs(raw, filtered_events, event_id={'Stimulus/s200':event_id['Stimulus/s200']}, tmin=-0.2, tmax=1.0, preload=True, 
                                    baseline=(None, 0), event_repeated='merge', verbose=False)
                epochs.drop_bad()
                return epochs

            epochs_c_noise = make_events(events_congr, raw_filtered)
            epochs_n_noise = make_events(events_not_congr, raw_filtered)
            reject_criteria = dict(eeg=130e-6)
            epochs_c = epochs_c_noise.copy().drop_bad(reject=reject_criteria, verbose=False)
            epochs_n = epochs_n_noise.copy().drop_bad(reject=reject_criteria, verbose=False)
            drop_log = epochs_c_noise.drop_log
            rejected_epochs = [i for i, log in enumerate(drop_log) if log]
            print(f'Удаленные эпохи C: {rejected_epochs}')
            drop_log = epochs_n_noise.drop_log
            rejected_epochs = [i for i, log in enumerate(drop_log) if log]
            print(f'Удаленные эпохи N: {rejected_epochs}')
            
            erp = {#'P1':['P3', 'Pz', 'P4', 'POz', 'P1', 'P2'],
                #'N1':['P3', 'Pz', 'P4', 'O1', 'Oz', 'O2'],
                #'CNV':['FCz', 'Cz'],
                #'N2':['P3', 'Pz', 'P4', 'POz', 'P1', 'P2'],
                'P3':['P3', 'Pz', 'P4', 'POz', 'P1', 'P2'],}
            fig, axs = plt.subplots(4, 2, figsize=(10, 15))  # Увеличенный размер для среднего графика
            for ax, electrode in zip(axs.flatten()[:-2], erp['P3']):
                mean_data_c = np.mean(epochs_c.average(picks=electrode, method='mean').data, axis=0)  
                mean_data_n = np.mean(epochs_n.average(picks=electrode, method='mean').data, axis=0) 
                sf = epochs_c.info['sfreq']  
                baseline = 0.2 
                t = np.arange(-baseline, len(mean_data_c)/sf - baseline, 1/sf)
                ax.plot(t, mean_data_c, label='con', color='blue')
                ax.plot(t, mean_data_n, label='n-con', color='red')
                ax.set_ylim([-10*1e-6, 10*1e-6])
                ax.set_title(f'ERP {electrode}')
                ax.set_xlabel('Время (с)')
                ax.set_ylabel('Амплитуда')
                ax.axhline(0, color='black', linewidth=1)
                ax.axvline(0, color='black', linewidth=1)  
                ax.grid(True)
                ax.legend()
            mean_data_c_all = np.mean([epochs_c.average(picks=electrode, method='mean').data for electrode in erp['P3']], axis=0)
            mean_data_n_all = np.mean([epochs_n.average(picks=electrode, method='mean').data for electrode in erp['P3']], axis=0)
            t = np.arange(-baseline, len(mean_data_c_all[0])/sf - baseline, 1/sf)
            axs[-1,0].plot(t, np.mean(mean_data_c_all, axis=0), label='con', color='blue', linestyle='--')
            axs[-1,0].plot(t, np.mean(mean_data_n_all, axis=0), label='n-con', color='red', linestyle='--')
            axs[-1,0].set_ylim([-10*1e-6, 10*1e-6])
            axs[-1,0].set_title(f'ERP {mode} Среднее по электродам, C-{len(epochs_c)}/{len(epochs_c_noise)}, N-{len(epochs_n)}/{len(epochs_n_noise)}')
            axs[-1,0].set_xlabel('Время (с)')
            axs[-1,0].set_ylabel('Амплитуда')
            axs[-1,0].axhline(0, color='black', linewidth=1)  
            axs[-1,0].axvline(0, color='black', linewidth=1)  
            axs[-1,0].grid(True)
            axs[-1,0].legend()
            plt.tight_layout()
            plt.show()
            fig.savefig(f'P3-{mode}.png')
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
    def __init__(self, montage, elc_file=None, mode='Cz', vis=False, threshold=0.08):
        self.montage = montage
        self.elc_file = elc_file
        self.vis = vis
        self.mode = mode
        self.threshold = threshold
    def forward(self, raw):
        if self.montage=='waveguard64':
            montage = create_custom_montage(self.montage)
        elif self.montage == 'personal':
            ch_dict, nasion, lpa, rpa, hsp = read_elc(self.elc_file)
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