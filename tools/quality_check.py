import numpy as np 
import mne 
from .savers import bridge_save, event_saver

def event_check(raw, saving_dir=None, vis=False):
    proc_count = 0
    mind_count = 0
    other_count = 0
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    rev_event_id = dict(list(zip(event_id.values(), event_id.keys())))
    mind_stimulus = ['Stimulus/s50', 'Stimulus/s52', 'Stimulus/s54', 'Stimulus/s56', 'Stimulus/s58',
                    'Stimulus/s60', 'Stimulus/s62', 'Stimulus/s64', 'Stimulus/s66', 'Stimulus/s68',
                    'Stimulus/s70', 'Stimulus/s72', 'Stimulus/s74', 'Stimulus/s76', 'Stimulus/s78', 'Stimulus/s80']
    
    proc_stimulus = ['Stimulus/s0', 'Stimulus/s1', 'Stimulus/s2', 'Stimulus/s3', 'Stimulus/s4', 'Stimulus/s9', 'Stimulus/s10']
    mind_counter = {stimulus: 0 for stimulus in mind_stimulus}
    proc_counter = {stimulus: 0 for stimulus in proc_stimulus}
    strange_list = set()
    for event in events:
        if rev_event_id[event[2]] in proc_counter.keys():
            for key in proc_counter.keys():
                if rev_event_id[event[2]] == key:
                    proc_counter[key]+=1
                    proc_count += 1
        elif rev_event_id[event[2]] in mind_counter.keys():
            for key in mind_counter.keys():
                if rev_event_id[event[2]] == key:
                    mind_counter[key]+=1
                    mind_count += 1
        else:
            strange_list.add(rev_event_id[event[2]])
            other_count += 1
    event_saver(proc_counter, mind_counter, saving_dir, vis)

def bridging_test(raw, saving_dir=None, vis=False):
    raw = raw.copy()
    bridged_idx, ed_matrix = mne.preprocessing.compute_bridged_electrodes(raw, verbose=False)
    correlation_matrix = np.corrcoef(raw.get_data())
    bridge_save(raw, correlation_matrix, bridged_idx, ed_matrix, saving_dir, vis=vis)
    return bridged_idx

def detect_noise_channel(raw, level=2):
    raw = raw.copy()
    data = raw.get_data()
    mean_row = np.mean(data, axis=0)[np.newaxis, :]
    data_c = data - mean_row
    std_devs = np.std(data_c, axis=1)
    threshold = np.mean(std_devs) + level * np.std(std_devs)
    noisy_channels = np.where(std_devs > threshold)[0]
    noisy_channels_names = np.array(raw.ch_names).take(noisy_channels)
    score = np.array(std_devs).take(noisy_channels)
    return noisy_channels_names, score

def detect_noise_channel_neigbours(raw, level=0.5):
    raw = raw.copy()
    data = raw.get_data()
    data_c = data.copy()
    adjacency, ch_names = mne.channels.find_ch_adjacency(info=raw.info, ch_type='eeg')
    std_devs = np.std(data, axis=1)
    normalized_std_devs = np.zeros(data.shape[0])
    
    bad_channels = []
    score = []
    for i, ch in enumerate(ch_names):
        neighbors = np.where(adjacency[i].toarray().flatten())[0]
        if len(neighbors) == 0:
            continue
        data_c[neighbors] = data[neighbors] - np.mean(data[neighbors], axis=0)[np.newaxis, :]
        std_devs[neighbors] = np.std(data_c[neighbors], axis=1)
        normalized_std_devs[neighbors] = (std_devs[neighbors] - np.mean(std_devs[neighbors])) / np.std(std_devs[neighbors])
        if normalized_std_devs[i] > level:
            bad_channels.append(i)
            score.append(normalized_std_devs[i])
    noisy_channels_names = np.array(raw.ch_names).take(bad_channels)
    return noisy_channels_names, score 

def analyze_component_properties(raw, ica, comps):
    def compute_psd(signal, sampling_rate):
        raw = mne.io.RawData(signal, sampling_rate)
        raw.compute_psd()
        return psd
    def psd_plot(psd):
        psd.plot()
    sfreq = raw.info['sfreq']
    data = ica.get_components()
    for comp in comps:
        freqs, psd = compute_psd(data[:, comp], sfreq)
        psd_plot(freqs, psd)
