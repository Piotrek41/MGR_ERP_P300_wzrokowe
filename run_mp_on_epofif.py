#!/usr/bin/python
# coding: utf8

import os
import sys
import json
import pandas as pd

import multiprocessing
import tempfile

from shellescape import quote

import numpy as np

import mne


from scipy import signal

from obci.analysis.obci_signal_processing.read_manager import ReadManager
from obci.analysis.obci_signal_processing.signal.read_data_source import MemoryDataSource
from obci.analysis.obci_signal_processing.signal.read_info_source import MemoryInfoSource


chnls_to_clusterize = {"local": ('F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'),
                       "global": ('F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'),
                       "vep": ('O1', 'Oz', 'O2', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'),
                       "p300-czuciowe": ('F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'),
                       "p300-sluchowe-slowa": ('F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'),
                       "p300-sluchowe-tony": ('F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'),
                       "p300-wzrokowe": ('O1', 'Oz', 'O2', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'),
                       "n400-slowa": ('Fp1', 'Fpz', 'Fp2','F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4'),
                       }

default_chnls_to_draw = ('Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4',
                         'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2', 'M1', 'M2')
                         
this_folder = os.path.abspath(os.path.dirname(__file__))

def n_parent_dirpath(path, n):
    """Funckja zwracająca n-ty w hierarchii macierzysty katalog, w którym leży plik.
    :param path: ścieżka do pliku/katalogu
    :param n: o ile stopni w hierarchii w górę podać katalog (1 to katalog, w którym leży plik, 2 to katalog, w którym leży katalog, w którym leży plik, itd.)
    :return: scieżka do n-tego w hierachii macierzystego katalogu
    """
    for i in range(n):
        path = os.path.dirname(path)
    return path

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

def generate_mp_book_empi_v1(x, path_to_book,book_name, mp_type,sampling_frequency):

    tmp_dir = tempfile.mkdtemp()
    x.astype('float32').tofile(os.path.join(tmp_dir, 'signal.bin'))

    [sample_count, channel_count] = x.shape
    MP_EPOCH_IN_SECONDS = sample_count/sampling_frequency


    if 'mmp' in mp_type:
        suffix = 'mmp'
    else:
        suffix = 'smp'
    
    # maximum_iterations = int(MP_EPOCH_IN_SECONDS * 5 * 2) 
    maximum_iterations = 15
    cpu_threads = multiprocessing.cpu_count()
    if mp_type == 'mmp1':
        mode = ' --mmp1'
    if mp_type == 'mmp3':
        mode = ' --mmp3'
    if mp_type == 'smp':
        mode = ''

    segment_size = int(sample_count)

    empi_params = "-c {} -f {} -i {} -r 0.01{} --segment-size {} --gabor --cpu-threads 1 --cpu-workers {} --energy-error=0.05 --gabor-scale-max 10 --gabor-freq-max 45.0 --gabor-scale-min 0.1 -o global".format(channel_count, sampling_frequency, maximum_iterations, mode, segment_size, cpu_threads)
    mp_path = os.path.join(this_folder, 'empi-lin64')
  
    invocation = 'cd ' + quote(
            tmp_dir) + ' && {} {} signal.bin {}.db&& mv {}.db '.format(quote(mp_path), empi_params, book_name, book_name) + quote(
        path_to_book)
    print("running:")
    print(invocation)
    if os.system(invocation):
        raise Exception('empi invocation failed')
    return empi_params

def reading_epoch_data_from_smart_tags(tags, chnames, bas = -0.1):
    """
    Args:
        tags: smart tag list - channel epochs from epofif,
        chnames: list of channels to use for averaging,
        bas: baseline (in negative seconds),

        channels_data: (tags,epochs,channels,samples) - example (100:21:819)  """

    min_length = min(i.get_samples().shape[1] for i in tags)
    # really don't like this, but epochs generated by smart tags can vary in length by 1 sample
    channels_data = []
    Fs = float(tags[0].get_param('sampling_frequency'))
    for i in tags:
        try:
            data = i.get_channels_samples(chnames)[:, :min_length]
        except IndexError:  # in case of len(chnames)==1
            data = i.get_channels_samples(chnames)[None, :][:, :min_length]
        
        if bas:
            for nr, chnl in enumerate(data):
                data[nr] = chnl - np.mean(chnl[0:int(-Fs * bas)])  # baseline correction
        if np.max(np.abs(data)) < np.inf:
            channels_data.append(data)
        
    
    return channels_data

def reshapeing_data_for_mp(epoch_data,sampling_freq,resampling_freq, resample = 0): 
    """
    Args:
        epoch_data: data from function "reading_epoch_data_from_smart_tags" """   

    epoch_data_shape = np.array(epoch_data).shape #(tags,epochs,channels,samples) - example (2:100:21:819)
    print('signal data shape(tags,number of epochs, number of channels, samples):',epoch_data_shape)
    #reshaping epoch_data to fit mmp1 algorithm - treat all epoch as an array (all_epochs:samples) - get two arrays for tags
    epoch_data_for_mp = []
    if resample == 0:
        temp=[]
        for chnl in range(epoch_data_shape[1]):
            for epoch in range(epoch_data_shape[0]):
                temp.append(epoch_data[epoch][chnl])
        epoch_data_for_mp.append(np.array(temp).T)
    else:
        num_of_desired_samples = int(epoch_data_shape[-1]/sampling_freq*resampling_freq)
        temp=[]
        for chnl in range(epoch_data_shape[1]):
            for epoch in range(epoch_data_shape[0]):
                s = signal.resample(epoch_data[epoch][chnl], num_of_desired_samples)
                temp.append(s)                
        epoch_data_for_mp.append(np.array(temp).T)        

    print('reshaped signal data(tags,samples,all_epochs_sorted_by_channels):',np.array(epoch_data_for_mp[0]).shape)

    '''all_epochs_sorted_by_channels - e.g. first 100 epochs belong to channel 1'''   

    return epoch_data_for_mp[0]

def run_mp(filelist,chnls_to_clusterize,default_chnls_to_draw,row=None, last_n=None, first_n=None):

    #######################reading data from *.fif files##########################
    
    fifpath_list = [filepath for filepath in filelist if "-epo.fif" in filepath]

    epochs_list = []
    if not fifpath_list:
        print('Please provide correct path to *-epo.fif')
        # exit()

    for fifpath in fifpath_list:
        try:
            mne_epochs = mne.read_epochs(fifpath)

            rm_list = mne_to_rm_list(mne_epochs)
            

            epochs_list.append( rm_list)

            epoch_labels = sorted(mne_epochs.event_id.keys(), key = lambda e: mne_epochs.event_id[e])
            # print("Using clean epochs from file {}".format(fifpath))
        except IOError:
            print('Please provide correct path to *-epo.fif')


    if last_n is not None:
        for i in range(len(epochs_list)):
            epochs_list[i] = epochs_list[i][-last_n:]

    if first_n is not None:
        for i in range(len(epochs_list)):
            epochs_list[i] = epochs_list[i][:first_n]

###############################setting file names and directories###########################################################

    if row != 28 and row != 48 and row != 10:
        full_commonprefix = os.path.commonprefix(fifpath_list)
        if "mont-" in full_commonprefix:
            commonprefix = full_commonprefix[:full_commonprefix.index("_mont-")]
        else:
            commonprefix = full_commonprefix
    else:
        commonprefix = [os.path.basename(i).split('.obci')[0] for i in fifpath_list][0]
        full_commonprefix=''
    # print(commonprefix)


    paradigm_name = os.path.basename(n_parent_dirpath(fifpath_list[0], 4))
    
    results_filename = paradigm_name + "_" + os.path.basename(commonprefix)

    montages = set()
    btypes = set()
    for fifpath in fifpath_list:
        pos_mont = fifpath.index("mont-")
        pos_btype = fifpath.index("btype-")

        pos_h = fifpath.index("h-")
    
        montage = fifpath[pos_mont + 5: pos_btype - 1]
        btype = fifpath[pos_btype + 6: pos_h - 1]

        if 'dd' in fifpath:
            btype += '_dd'

        if 'od' in fifpath:
            btype += '_od'
        montages.add(montage)
        btypes.add(btype)


    montages = list(montages)
    btypes = list(btypes)    
    if "_h-" in full_commonprefix:
        pos_h = fifpath.index("_h-")
        h = full_commonprefix[pos_h + 3: pos_h + 3 + 8]
        results_filename = os.path.basename(results_filename) + '_{}_{}_{}'.format(montages, btypes, h)
    else:
        results_filename = os.path.basename(results_filename) + '_{}_{}'.format(montages, btypes)
    outdir = os.path.join(os.path.dirname(os.path.dirname(fifpath_list[0])), "mp_books")

    try:
        os.makedirs(outdir)
    except OSError:
        pass
################################converting data into lists/arrays##########################################################

    # chnames = [i for i in default_chnls_to_draw if 'F' not in i and 'T3' not in i and 'T4' not in i] 
    chnames = default_chnls_to_draw
    # for tag in epochs_list[0][0] + epochs_list[0][1]: #epochlist
    #     available_chnls = tag.get_param('channels_names')
    #     Fs = float(tag.get_param('sampling_frequency'))
    #     chnames = [chname for chname in chnames if chname in available_chnls]
    #     chnls_to_clusterize = [chname for chname in chnls_to_clusterize if chname in available_chnls]
    chn=[]    
    chn_id=[]
    for i in range(len(epochs_list)):

        available_chnls = epochs_list[i][0][0].get_param('channels_names')
        Fs = float(epochs_list[i][0][0].get_param('sampling_frequency'))
        chnames = [chname for chname in chnames if chname in available_chnls] 
        chn.append(chnames)
        chn_id.append(len(chnames))


    start_offset = mne_epochs.tmin  
    chnames = chn[np.argmin(chn_id)]
    # print(chnames)
    epoch_data = []
    for i in range(len(epochs_list)):
  
        for tags in epochs_list[i]:
            ev= reading_epoch_data_from_smart_tags(tags, chnames, start_offset)
            epoch_data.append(np.array(ev))  
      
    # # epoch_data = [np.concatenate((epoch_data[0],epoch_data[2]),axis=0),np.concatenate((epoch_data[1],epoch_data[3]),axis=0)]
    epoch_data = [np.concatenate([epoch_data[i] for i in range(0,len(epoch_data),2)],axis=0), np.concatenate([epoch_data[i] for i in range(1,len(epoch_data),2)],axis=0)]
    # # print(epoch_data[0].shape,epoch_data[1].shape)
    ##############################################################################

    # '''epoch_data is a lis of shape: e.g. (2:100:21:819) or [(90:21:819),(100,21,819)] so be careful when using np.array(channels_data) because 90 and 100 are of different sizes'''
    # # print(np.array(epoch_data[0]).shape)


    resampling_freq =int(Fs/4)

    epoch_data_for_mp = []
    for tags in range(len(epoch_data)):
        ev= reshapeing_data_for_mp(epoch_data[tags],Fs,resampling_freq,resample = 1)
        epoch_data_for_mp.append(ev)


    # with open('/mnt/c/Users/Piotr/Desktop/pliki_pulpit/budzik/mp/choseing_atom/epoch_numbers.txt','a') as epoch_number:
    #     epoch_number.write('{}\n'.format(round(100*(np.array(epoch_data_for_mp[0].shape[1])/len(chnames))/(np.array(epoch_data_for_mp[1].shape[1])/len(chnames)),2)))
    
    # # print(epoch_data_for_mp.shape)
    # # print(len(epoch_data_for_mp),epoch_data_for_mp[0].shape)
    book_data = np.zeros((epoch_data_for_mp[0].shape[0],epoch_data_for_mp[0].shape[1]+epoch_data_for_mp[1].shape[1]))
    idx=0
    for i in epoch_data_for_mp:
        for j in range(i.shape[1]):
            book_data[:,idx]+=i[:,j]
            idx+=1
    # print(book_data.shape)

    # print(outdir)
    # print(os.path.basename(results_filename))

    # for tag_index,signal_data in enumerate(epoch_data_for_mp):
        # print(np.array(epoch_data_for_mp[tag_index].shape[1])/len(chnames),np.array(epoch_data_for_mp[tag_index]).shape[0])

    empi_params = generate_mp_book_empi_v1(book_data,outdir ,quote(os.path.basename(results_filename)), 'mmp1',resampling_freq)
   
        #creating text file with empi_params
    with open(os.path.join(outdir,os.path.basename(results_filename))+'_empi_params.txt', 'w') as fp:
        fp.write('EMPI PARAMS\n')
        for item_index, item in enumerate(empi_params.split('--')):
            
            if item_index == len(empi_params.split('--')) -1:
                tmp = item.split(' ')
                
                fp.write('--'+" ".join(tmp[:-2])+'\n')
                fp.write(" ".join(tmp[-2:])+'\n')
            elif item_index == 0:
                for i in item.split('-')[1:]:
                    fp.write('-'+i+'\n')
            else: 
                fp.write('--'+item+'\n')
                
        fp.write('\nDATA DESCRIPTION\n{} {}\nNumber of channels: {}\nNumber of epochs: {} {}\nNumber of samples: {}\n -c = Number of channels *Number of epochs\nEvery {} or {} -c belongs to a different channel\n{}'.format(
            epoch_labels[0],
            epoch_labels[1],
            len(chnames),

            np.array(epoch_data_for_mp[0].shape[1])/len(chnames),
            np.array(epoch_data_for_mp[1].shape[1])/len(chnames),

            np.array(epoch_data_for_mp[0]).shape[0],
            

            np.array(epoch_data_for_mp[0].shape[1])/len(chnames),
            np.array(epoch_data_for_mp[1].shape[1])/len(chnames),
            chnames))
   
        print('Done')      

    # with open('/mnt/c/Users/Piotr/Desktop/pliki_pulpit/budzik/mp/choseing_atom/db_names_no_frontal_electrodes.txt','a') as db_names:

    #     try:
    #         db_files = [i for i in os.listdir(outdir) if i.endswith('.db') and results_filename.split('[')[0] in i][0]
    #     except:
    #         db_files = [i for i in os.listdir(outdir) if i.endswith('.db') and results_filename in i][0]           
    #     # db_names.write('{}.db\n'.format(os.path.join(outdir,results_filename)))
    #     db_names.write('{}\n'.format(os.path.join(outdir,db_files)))
# if __name__ == '__main__':
#     if len(sys.argv) < 2:
#         raise IOError('Please provide path to *-epo.fif')
#     else:
#         run_mp(get_filelist(sys.argv[1:]), chnls_to_clusterize,default_chnls_to_draw)

if __name__ == '__main__':

######################this is an example of how I managed file merging based on the *.csv file shown below###############################
    data = pd.read_csv("/mnt/c/Users/Piotr/Desktop/pliki_pulpit/budzik/p300-wzrokowe/Measurements_database_merged.csv")
    data = data.loc[data['measurement_type']=='p300-wzrokowe']
    data=np.array(data[['person_id','diag','auc_wzrokowe','auc_wzrokowe_perc','fif_file_paths']])
    for i in data:
        i[-1]=[os.path.basename(j) for j in json.loads(i[-1])]
    
    fif_files = [i[-1] for i in data]
    files_fif =[]

    #tree of folders with *.fif files
    for path, subdirs, files in os.walk(sys.argv[1]):
        files_fif.extend([os.path.join(path, name) for name in files if name.endswith('.fif')])
    for i,dir in enumerate(fif_files):
        temp=[]
        for file in files_fif:
            if os.path.basename(file) in dir:
                temp.append(file)
        fif_files[i] = temp

    for row,paths_to_fif in enumerate(fif_files):

        #paths_to_fif --- is a list of *.fif files that will be merged into one data set, can be a list with path to one file only

        if row>=0:# and '/15/' in paths_to_fif[0] or '/21/' in paths_to_fif[0] or '/23/' in paths_to_fif[0] or '/16/' in paths_to_fif[0] or '/17/' in paths_to_fif[0] :# and len([i for i in paths_to_fif if '/01/' not in i and '/03/' not in i and '/04/' not in i and '/05/' not in i and '/07/' not in i and '/09/' not in i and '/10/' not in i and '/11/' not in i and '/12/' not in i and '/13/' not in i and '/14/' not in i])>0:
            try:
                print('Running calculations on {}'.format(paths_to_fif))

                run_mp(paths_to_fif, chnls_to_clusterize,default_chnls_to_draw,row=row)

            except Exception as e:

                # with open('/mnt/c/Users/Piotr/Desktop/pliki_pulpit/budzik/mp/choseing_atom/db_names_no_frontal_electrodes.txt','a') as db_names:
                #     db_names.write('[]\n')
                # with open('/mnt/c/Users/Piotr/Desktop/pliki_pulpit/budzik/mp/choseing_atom/epoch_numbers.txt','a') as epoch_number:
                #     epoch_number.write('{}\n'.format([]))                
                pass

 #u can fit os.walk function for yourself to get list with paths to file/s that u want to merge 
 # straight from the directory that u have your data stored

    # if len(sys.argv) < 2:
    #     raise IOError('Please provide path to *-epo.fif or the directory containing *-epo.fif files.')
    # else:
    #     i=0
    #     for path, subdirs, files in os.walk(sys.argv[1]):
    #         # for name in files:
    #         #     if '.fif' in name and '/p1/' not in path:
    #         #         path_to_epofif_file=os.path.join(path, name)
    #         #         print('Runing calculations on:\n',[os.path.join(path, name)],'\n')
    #         files_db =[os.path.join(path, name) for name in files if '.fif' in name]
    #         if len(files_db) >0:
    #             # print(files_db)
    #             i+=1
    #             if i >12:
    #                 run_mp([os.path.join(path, name) for name in files if '.fif' in name], chnls_to_clusterize,default_chnls_to_draw)     
