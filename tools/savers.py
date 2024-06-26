
import seaborn as sns
import matplotlib.pyplot as plt
import mne 
import pandas as pd 
import os 

def bridge_save(raw, correlation_matrix, bridged_idx, ed_matrix, saving_dir=None, vis=False):
    channels = raw.ch_names
    bridge_fig = mne.viz.plot_bridged_electrodes(raw.info, bridged_idx, ed_matrix)
    fig_corr = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(correlation_matrix, cmap='viridis', xticklabels=channels, yticklabels=channels) #np.mean(ed_matrix, axis=0)
    ax.set_xticklabels(channels, rotation=90, fontsize=7)
    ax.set_yticklabels(channels, rotation=0, fontsize=7)
    plt.title('Correlation of Channel Connections (R)')
    plt.xlabel('Channels')
    plt.ylabel('Channels')
    if saving_dir:
        plt.savefig(os.path.join(saving_dir, f'сorrelation_electrodes.png'))
        bridge_fig.savefig(os.path.join(saving_dir, f'bridged_electrodes.png'))
    if not vis:
        plt.close(fig_corr)
        plt.close(bridge_fig)

def components_save(raw, ica, exclude_idx, labels, probas, saving_dir=None, vis=False):
    duration = raw.times[-1]*1000
    start_time = 0 
    stop_time = min(10*1000, duration)  
    ica_fig = ica.plot_components(title='ICA Components', show=vis)
    if isinstance(raw, mne.io.Raw) or isinstance(raw,mne.io.brainvision.brainvision.RawBrainVision):
        overlay_fig = ica.plot_overlay(raw, title='Overlay', start=start_time, stop=stop_time, 
                                    exclude=exclude_idx, show=vis)
    else:
        overlay_fig = ica.plot_overlay(raw.average(), title='Overlay', start=start_time, stop=stop_time, 
                                    exclude=exclude_idx, show=vis)
    if saving_dir:
        ica_fig.savefig(os.path.join(saving_dir, 'ica_components.png'))
        overlay_fig.savefig(os.path.join(saving_dir, 'overlay.png'))
    if not vis:
        plt.close(ica_fig)
        plt.close(overlay_fig)
    for idx in exclude_idx:
        ica_exclude_fig = ica.plot_properties(raw, picks=idx, show=vis, verbose=False)[0]
        if saving_dir:
            if labels is None:
                ica_exclude_fig.savefig(os.path.join(saving_dir, f'ica_exclude_components_{idx}_label:hand.png'))
            else:
                ica_exclude_fig.savefig(os.path.join(saving_dir, f'ica_exclude_components_{idx}_{labels[idx]}_{probas[idx]}.png'))
        if not vis:
            plt.close(ica_exclude_fig)

def event_saver(proc_counter, mind_counter, saving_dir=None, vis=False):
    df = pd.read_csv('tools/ASSR.csv', index_col=None)
    combined_dict = {**proc_counter, **mind_counter}
    combined_dict = {key.replace('Stimulus/', ''): value for key, value in combined_dict.items()}
    df.columns = df.columns.str.replace(' ', '')
    new_row = {column: combined_dict[column] for column in df.columns}
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)
    target_row = df.iloc[0].copy()
    target_row_numeric = pd.to_numeric(target_row, errors='coerce')
    new_row_numeric = pd.to_numeric(pd.Series(new_row), errors='coerce')
    difference_row = target_row_numeric - new_row_numeric
    difference_row_df = pd.DataFrame([difference_row])
    df = pd.concat([df, difference_row_df], ignore_index=True)

    df.insert(0, 'Cond', ['Target', 'Real', 'Diff'])
    if saving_dir:
        df.to_csv(os.path.join(saving_dir, 'stimulus_counts.csv'), index=False)
    fig_table, ax = plt.subplots(figsize=(12, 2))  
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title('Таблица встречаемости стимулов', fontsize=14, pad=1)
    if saving_dir:
        fig_table.savefig(os.path.join(saving_dir, 'stimulus_counts.png'), bbox_inches='tight', pad_inches=0.1)
    if not vis:
        plt.close(fig_table)

def tfr_plot(tfr, itc, picks, fmin, fmax, vmin=None, vmax=None, baseline=(None, 0), combine='mean', 
             baseline_mode='logratio', saving_dir=None, vis=False, 
             tfr_title='Event Related Spectral Perturbation (ERSP)',
             itc_title='Inter-Trial Coherence (ITC)'):  
    tfr_fig = tfr.plot(title=tfr_title, fmin=fmin, fmax=fmax, vlim=(vmin,vmax), 
             picks=picks, combine=combine, baseline=baseline, mode=baseline_mode)[0] 
    itc_fig = itc.plot(title=itc_title, fmin=fmin, fmax=fmax, 
             picks=picks, combine=combine, baseline=baseline, mode=baseline_mode)[0]
    if not vis:
        plt.close(tfr_fig)
        plt.close(itc_fig)
    if saving_dir:
        tfr_fig.savefig(os.path.join(saving_dir, f'ERSP.png'))
        itc_fig.savefig(os.path.join(saving_dir, f'ITC.png'))