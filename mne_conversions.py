# coding: utf8
import os.path
import mne
import numpy as np
from obci.analysis.obci_signal_processing.read_manager import ReadManager
from obci.analysis.obci_signal_processing.signal.read_data_source import MemoryDataSource
from obci.analysis.obci_signal_processing.signal.read_info_source import MemoryInfoSource


def mne_to_rm_list(e_mne):
    epochs_l = []
    for n, s in enumerate(sorted(e_mne.event_id.values())):
        mask = e_mne.events[:, 2] == s
        epochs = e_mne[mask].get_data() * 1e6
        epochs_type = []
        for epoch in epochs:
            mds = MemoryDataSource(epoch)
            mis = MemoryInfoSource(p_params = {'sampling_frequency': str(e_mne.info['sfreq']),
                                               'channels_names': e_mne.ch_names,
                                               'number_of_channels': len(e_mne.ch_names),
                                               'file': e_mne.info['description']
                                               })
            rm = ReadManager(p_data_source = mds, p_info_source = mis, p_tags_source = None)
            epochs_type.append(rm)
        epochs_l.append(epochs_type)
    return epochs_l


def mne_info_from_rm(rm):
    chnames = rm.get_param('channels_names')
    sfreq = float(rm.get_param('sampling_frequency'))
    ch_types = [chtype(i) for i in chnames]
    info = mne.create_info(ch_names = chnames, sfreq = sfreq, ch_types = ch_types)#, montage = 'standard_1005')
    # info['description'] = os.path.basename(rm.get_param('file'))
    return info


def read_manager_to_mne(epochs, baseline = None, epoch_labels = None):
    """Returns all epochs in one mne.EpochsArray object and slices for every tagtype
    :param epochs: list of epochs (rm Smart Tags)
    :param baseline: length of baseline before start of an epoch
    :param epoch_labels: list of description of epoch types (len 2 if two types)
    :return:
    """
    all_epochs = []
    tag_type_slices = []
    last_ep_nr = 0
    for tagtype_e in epochs:
        for epoch in tagtype_e:
            all_epochs.append(epoch.get_samples() * 1e-6)
        len_tagtype = len(tagtype_e)
        tag_type_slices.append(slice(last_ep_nr, len_tagtype + last_ep_nr))
        last_ep_nr = len_tagtype + last_ep_nr
    
    info = mne_info_from_rm(epoch)
    
    min_length = min(i.shape[1] for i in all_epochs)
    
    all_epochs = [i[:, 0:min_length] for i in all_epochs]
    all_epochs_np = np.stack(all_epochs)
    
    event_types = np.ones((len(all_epochs), 3), dtype = int)
    event_types[:, 0] = np.arange(0, len(all_epochs), 1)
    for n, s in enumerate(tag_type_slices):
        event_types[s, 2] = n
    
    print(info)
    if epoch_labels:
        e_mne = mne.EpochsArray(all_epochs_np, info, tmin = baseline, baseline = (baseline, 0), events = event_types,
                                event_id = {epoch_labels[0]: 0, epoch_labels[1]: 1})
    else:
        e_mne = mne.EpochsArray(all_epochs_np, info, tmin = baseline, baseline = (baseline, 0), events = event_types)
    
    return e_mne, tag_type_slices


def nparrays_to_mne(target_epochs, nontarget_epochs, outdir, patient, info = None, blok_type = 1, montage = ['car'], chnames = None, plot_only = False):
    """Przykład użycia:
    import mne_conversions as mc
    import scipy.stats as st
    T = st.norm.rvs(size = (20, 21, 1126))
    NT = st.norm.rvs(size = (80, 21, 1126))
    mc.nparrays_to_mne(T, NT, "/repo/fizmed/coma/results/simulated-test", "S01")"""
    
    
    all_epochs = []
    for epoch in target_epochs:
        all_epochs.append(epoch)
    for epoch in nontarget_epochs:
        all_epochs.append(epoch)

    if chnames is None:
        chnames = ("Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "Oz", "O2")
    sfreq = 1024.
    ch_types = ['eeg']*len(chnames)
    if info is None:
        info = mne.create_info(ch_names = chnames, sfreq = sfreq, ch_types = ch_types, montage = 'standard_1005')
        info['description'] = patient
    
    min_length = min(i.shape[1] for i in all_epochs)
    
    all_epochs = [i[:, 0:min_length] for i in all_epochs]
    all_epochs_np = np.stack(all_epochs)
    
    event_types = np.ones((len(all_epochs), 3), dtype = int)
    event_types[:, 0] = np.arange(len(all_epochs))
    event_types[:target_epochs.shape[0], 2] = 0
    event_types[target_epochs.shape[0]:, 2] = 1
    
    e_mne = mne.EpochsArray(all_epochs_np, info, tmin = -0.1, baseline = (-0.1, 0), events = event_types, event_id = {'target': 0, 'nontarget': 1})

    h = str(hex(hash(patient + str(list(target_epochs)) + str(list(nontarget_epochs))
        )))[-8:]
    fif_filename = patient + '_mont-{}_btype-{}_h-{}_clean_epochs-epo.fif'.format(montage, blok_type, h)
    
    fif_filepath = os.path.join(outdir, "p300-simulated", patient, "tura_1", "clean_epochs", fif_filename)
    
    directory = os.path.dirname(fif_filepath)
    try:
        os.makedirs(directory)
    except OSError:
        pass
    
    if plot_only:
        e_mne.plot_psd(show = False)
        e_mne.plot(scalings = {'eeg': 4e-5, 'eog': 4e-5}, show = True, block = True)
    else:
        e_mne.save(fif_filepath)


def read_manager_continious_to_mne(rm):
    info = mne_info_from_rm(rm)
    raw = mne.io.RawArray(rm.get_samples() * 1e-6, info)
    return raw


def chtype(name):
    ineeg = ('Fp1', 'Fpz', 'Fp2',
             'F7', 'F3', 'Fz', 'F4', 'F8',
             'M1', 'T3',
             'C3', 'Cz', 'C4',
             'T4', 'M2',
             'T5',
             'P3', 'Pz', 'P4',
             'T6',
             'O1', 'Oz', 'O2')
    ineog = ('eog', 'EOG',
             'Fpx')

    if name in ineeg:
        return 'eeg'
    elif name in ineog:
        return 'eog'
    else:
        return 'misc'
