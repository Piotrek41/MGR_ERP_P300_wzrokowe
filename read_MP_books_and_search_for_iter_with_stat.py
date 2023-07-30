import os
import numpy as np
import sqlite3

import mne
import sys

import scipy.stats as stats

# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as pb
import pylab as pb
import pandas as pd

from pandas import DataFrame
from tqdm.contrib.concurrent import process_map
from functools import partial

from multiprocessing import cpu_count

from statsmodels.sandbox.stats.multicomp import multipletests
# from plot_dipol import show_example,plot_image
def read_db_atoms_dipol(atom_db_path):
    if os.path.exists(atom_db_path) and atom_db_path.endswith('db'):
        cursor = sqlite3.connect(atom_db_path).cursor()
        # need to create a txt compatable numpy array
        # channel, iteration, modulus, amplitude (not p2p), position (absolute), width, frequency phase
        segment_length = float(cursor.execute(
            'SELECT segment_length_s FROM segments'
        ).fetchone()[0])
        atoms_db = cursor.execute(
            "SELECT channel_id, iteration, energy, amplitude, t0_s, scale_s, f_Hz, phase, segment_id FROM atoms where envelope='gauss'"
        )
        atoms = np.array(atoms_db.fetchall())

        ################# setting right channels and segements(epchs) ids ####################
        # empi_params_files= [book_file for book_file in os.listdir(os.path.dirname(atom_db_path)) if book_file.endswith(".txt")]
        # if 'dd' in os.path.basename(atom_db_path):
        #     empi_params_files = [i for i in empi_params_files if 'dd' in i]
        # else:
        #     empi_params_files = [i for i in empi_params_files if 'od' in i]

        # bf=[book_file for book_file in os.listdir(os.path.dirname(atom_db_path)) if book_file.endswith(".txt")]
        # print('_'.join(bf[0].split('_')[:-2]))
        # exit()
        try:
            empi_params_files= [book_file for book_file in os.listdir(os.path.dirname(atom_db_path)) if book_file.endswith(".txt") and os.path.basename(atom_db_path)[:-3] == '_'.join( book_file.split('_')[:-2])]
        except:
            empi_params_files= [book_file for book_file in os.listdir(os.path.dirname(atom_db_path)) if book_file.endswith(".txt")]
        try:
            empi_params_file = [i for i in empi_params_files][0]
        except:
            empi_params_file = [i for i in empi_params_files][1]
        print(os.path.basename(atom_db_path)[:-3])
        print(empi_params_file)

        with open(os.path.join(os.path.dirname(atom_db_path),empi_params_file) ) as f:
            contents = f.readlines()
            
            contents = [i.strip('\n') for i in contents]
            channel_names =[str(i[1:-1]) for i in contents[-1].strip('[]').split(', ')]
            contents = contents[:-3]

        [nb_channels,nb_epochs] = [i.split(' ')[-2:] for i in contents if 'Number of epochs' in i or 'Number of channels' in i ]
        nb_channels = int(nb_channels[-1])
        nb_epochs=[int(float(i)) for i in nb_epochs]
        
        iterations = [int(float(i.split(' ')[-2])) for i in contents if '-i' in i][0]
        labels = [contents[i+1] for i,x in enumerate(contents) if 'DATA DESCRIPTION' in x][0].split(' ')
        # print(labels,iterations,nb_channels,nb_epochs)


        atoms=np.array(atoms)
        # print(atoms[iterations*nb_channels*nb_epochs[0]:,0].shape,np.array([[i]*iterations for i in np.arange(nb_channels*nb_epochs[1])]).shape)
        # exit()
        atoms[iterations*nb_channels*nb_epochs[0]:,0]=np.array([[i]*iterations for i in np.arange(nb_channels*nb_epochs[1])]).reshape(iterations*nb_channels*nb_epochs[1])
        fs = int(float(contents[2].split(' ')[-2]))

        macroatom_id_1 =np.array([[i]*iterations for i in np.arange(nb_epochs[0])]*nb_channels)
        macroatom_id_1=np.reshape(macroatom_id_1,(macroatom_id_1.shape[0]*macroatom_id_1.shape[1],))
        atoms[:iterations*nb_channels*nb_epochs[0],-1]= macroatom_id_1

        macroatom_id_2 =np.array([[i]*iterations for i in np.arange(nb_epochs[1])]*nb_channels)
        macroatom_id_2=np.reshape(macroatom_id_2,(macroatom_id_2.shape[0]*macroatom_id_2.shape[1],))
        atoms[iterations*nb_channels*nb_epochs[0]:,-1]= macroatom_id_2

        channels = np.array([[i]*nb_epochs[0]*iterations for i in range(nb_channels)])
        channels=np.reshape(channels,(channels.shape[0]*channels.shape[1],))
        atoms[:iterations*nb_channels*nb_epochs[0],0]= channels

        channels2 = np.array([[i]*nb_epochs[1]*iterations for i in range(nb_channels)])
        channels2=np.reshape(channels2,(channels2.shape[0]*channels2.shape[1],))
        atoms[iterations*nb_channels*nb_epochs[0]:,0]= channels2

        return atoms[:iterations*nb_channels*nb_epochs[0]],atoms[iterations*nb_channels*nb_epochs[0]:],channel_names,fs,segment_length,labels #returns array with atom information from db [channel_id, iteration, energy, amplitude, t0_s, scale_s, f_Hz, phase, segment_id]
    else:
        raise FileNotFoundError(atom_db_path) 



def get_atoms(atoms, mp_params, width_coeff=1):

    columns = ['iteration', 'energy', 'amplitude', 'width', 'frequency', 'phase', 'struct_len', 'absolute_position', 'offset',
               'ch_id', 'ch_name','macroatom_id']

    dtypes = {'iteration': int, 'energy': float, 'amplitude': float,
              'width': float, 'frequency': float, 'phase': float,
              'struct_len': float,
              'absolute_position': float,
              'offset': float,
              'ch_id': int,
              'ch_name': str,
              'macroatom_id':int,

    }

    chosen = []
    
    for atom_id, atom in enumerate(atoms):
        # [channel_id, iteration, energy, amplitude, t0_s, scale_s, f_Hz, phase, segment_id]
        channel = int(atom[0])
        iteration = atom[1]
        energy = atom[2]
        amplitude = atom[3]
        position = atom[4]
        width = atom[5]
        frequency = atom[6]
        phase = atom[7]
        macroatom_id = atom[8]
        struct_len = width_coeff * width


        offset = position - struct_len / 2
        mp_chnames = mp_params
        mp_channel_id = channel 
        
        mp_channel_name = mp_chnames[mp_channel_id]

        chosen.append([iteration, energy,
                       amplitude, width, frequency, phase, struct_len, position, offset,
                       channel, mp_channel_name,macroatom_id
                       ])

    if not chosen:
        df = DataFrame(columns=columns)
    else:
        df = DataFrame(chosen, columns=columns)

    df = df.astype(dtype=dtypes)

    return df


def groups(group):
    example = group[1]
    return example

def gabor(t, s, t0, f0, phase=0.0, segment_length_s=20.0):
    """
    Generates values for Gabor atom with unit amplitude.
    t0 - global, t - also global,
    """

    t_segment = t % segment_length_s
    t0_segment = t0 % segment_length_s
    result = np.exp(-np.pi*((t_segment-t0_segment)/s)**2) * np.cos(2*np.pi*f0*(t_segment-t0_segment) + phase)
    return result


#############setting up data for topomaps and later plots######################

# #['iteration', 'modulus', 'amplitude', 'width', 'frequency', 'phase', 'struct_len', 'absolute_position', 'offset','ch_id', 'ch_name','macroatom_id']
def get_macroatom_reconstrucions(atoms_df,channel_names,fs,segment_length,label):

    macroatom_reconstrucions = []

    topomap_params = process_map(groups, atoms_df.groupby(by='iteration'),desc='Getting {} macroatom reconstruction'.format(label), chunksize=1)
    t=np.linspace(0,segment_length,int(fs*segment_length))
    chosen_iterations = []
    for df_i,df in enumerate(topomap_params):
        # i = 0
        # count = 0
        temp = []
        sigs = []
        for id, atom in df.iterrows():
            # if 0.2<atom['absolute_position']<0.9:
            if atom['frequency']<7 and 0.2<atom['absolute_position']<0.95:
            # if atom['frequency']<4 and 0.2<atom['absolute_position']<0.95:
                temp.append(1) 

            sig = gabor(t, atom['width'],atom['absolute_position'], atom['frequency'], atom['phase'], segment_length)
            sig*= atom['amplitude']
            channel = int(atom['ch_id'])
            sigs.append(sig) 
            # temp.append(sig)   
                # count+=1

            # i+=1  
        
        if np.sum(temp)!=0:
            
            chosen_iterations.append(df_i)
        macroatom_reconstrucions.append(sigs)
    
    macroatom_reconstrucions =np.array(macroatom_reconstrucions)

    macroatom_reconstrucions=macroatom_reconstrucions.reshape((macroatom_reconstrucions.shape[0],len(set(atoms_df['ch_id'])),len(set(atoms_df['macroatom_id'])),macroatom_reconstrucions.shape[-1]))

    macroatom_reconstrucions_mean =np.mean(macroatom_reconstrucions,axis=2)
    evoked = []
    for j in range(macroatom_reconstrucions_mean.shape[1]):
        evoked_atoms = []
        for i in range(macroatom_reconstrucions_mean.shape[0]):
            evoked_atoms.append(macroatom_reconstrucions_mean[i][j])

            # pb.plot(t,macroatom_reconstrucions[j][i],label=str(i))
        evoked.append(evoked_atoms)
        # pb.legend()
        # pb.show()
    evoked=np.array(evoked)#* 1e-6    
    return macroatom_reconstrucions, evoked, channel_names, fs, segment_length, chosen_iterations


def get_atoms_amplitude_for_test(group,ref_channel='Oz',mp_params=[]):#mp_params=[channel_names,fs],ref_channel='average'
    """group - groupby pandas object, gives a tuple (nr, pandas sub dataframe)"""
    channels = mp_params[0]
    fs=mp_params[1]
    if ref_channel == 'average':
        ref_channels = []
    else:
        ref_channels = ref_channel.strip().split(',')
        if len(ref_channels) == 1 and ref_channels[0] == '':
            ref_channels = []
    missing_channels = list(set(ref_channels) - set(channels))

    try:
        example = group[1]
    except:
        example=group

    missing_channels_count = len(missing_channels)
    data = np.array(list(zip(list(example.amplitude.values),list(example.energy.values))) + [0.0, ] * missing_channels_count)[:, None] #* 1e-6
    return data


def get_chosen_macroatom_reconstrucions(atoms_df, fs, segment_length,chosen_atom, amplitude=False,phase=False):

    macroatom_reconstrucions = []

    topomap_params = process_map(groups, atoms_df.groupby(by='iteration'), chunksize=1)
    t=np.linspace(0,segment_length,int(fs*segment_length))
    for df_i,df in enumerate(topomap_params):
        if df_i == chosen_atom:


            for id, atom in df.iterrows():
                if phase and len(phase)<=2:
                    if atom['phase'] <0:
                        # print('if')
                        phs = np.max(phase)
                    else:
                        # print('else')
                        phs = np.min(phase)
                    sig = gabor(t, atom['width'],atom['absolute_position'], atom['frequency'], phs, segment_length)
                elif phase and len(phase)>2:
                    phs = phase[atom['ch_id']]
                    sig = gabor(t, atom['width'],atom['absolute_position'], atom['frequency'], phs, segment_length)                   
                else:
                    sig = gabor(t, atom['width'],atom['absolute_position'], atom['frequency'], atom['phase'], segment_length)
                if amplitude:
                    sig*=amplitude
                else:
                     sig*=atom['amplitude']               

                macroatom_reconstrucions.append(sig)            
    
    macroatom_reconstrucions =np.array(macroatom_reconstrucions)

    macroatom_reconstrucions=macroatom_reconstrucions.reshape((len(set(atoms_df['ch_id'])),len(set(atoms_df['macroatom_id'])),macroatom_reconstrucions.shape[-1]))


  
    return macroatom_reconstrucions

def do_permutation_test(reshaped_signals,roi,chnames):
    clusters = []
    # for tagsnr in range(len(epoch_labels)):
            
    for nr, chname in enumerate(chnames):
        ######permutation test#########
        data_test = [reshaped_signals[0][nr, :, roi[0]:roi[1]],reshaped_signals[1][nr, :, roi[0]:roi[1]]]#type list;(tag,channels,epochs,samples)
        if len(data_test) > 1:
            T_obs, clusters_, cluster_p_values, H0 = mne.stats.permutation_cluster_test(data_test, step_down_p = 0.05, n_jobs = cpu_count()//2, seed = 42, out_type="mask")
        else:
            T_obs, clusters_, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(data_test[0], step_down_p = 0.05, n_jobs = cpu_count()//2, seed = 42, out_type="mask")
        clusters.append((clusters_, cluster_p_values)) 
    return clusters    

def sings_from_reconstructs(reconstructs):
    maxs =np.argmax(np.abs(reconstructs),axis =3)
    signs = []
    for ch in range(maxs.shape[1]):
        for e in range(maxs.shape[2]):
            for a in range(maxs.shape[0]):  
                amp = reconstructs[a,ch,e,maxs[a,ch,e]]
                if amp <0:
                    signs.append(-1)
                else:
                    signs.append(1)
    signs = np.array(signs) 
    return signs
def statistic(x, y, axis):
    return (np.mean(x, axis=axis) - np.mean(y, axis=axis) )/ (np.var(x, axis=axis) + np.var(y, axis=axis))**0.5
def main(atom_db_path,row, name, outdir,diag):
    atoms_target,atoms_nontarget,channel_names,fs,segment_length,labels = read_db_atoms_dipol(atom_db_path)

    #nontarget
    atoms_df = get_atoms(atoms_nontarget, channel_names, width_coeff=1)   

    macroatom_reconstrucions, evoked, channel_names, fs, segment_length, chosen_iterations =get_macroatom_reconstrucions(atoms_df,channel_names,fs,segment_length,labels[1])

    #target
    atoms_df2 = get_atoms(atoms_target, channel_names, width_coeff=1)  

    macroatom_reconstrucions2, evoked2, channel_names2, fs2, segment_length2, chosen_iterations2 = get_macroatom_reconstrucions(atoms_df2,channel_names,fs,segment_length,labels[0])#target

    signs_nt = sings_from_reconstructs(macroatom_reconstrucions) 
    signs_t = sings_from_reconstructs(macroatom_reconstrucions2)  
    # print(signs_nt.shape,signs_t.shape)


    p300_channels =('O1', 'Oz', 'O2', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4','T5','T6')

    
    t=np.linspace(0,segment_length,int(fs*segment_length))-0.2

    nontarget = DataFrame(np.swapaxes(np.vstack([np.array(atoms_df['amplitude'])*signs_nt,np.array(atoms_df['energy']),np.array(atoms_df['iteration']),np.array(atoms_df['macroatom_id']) ]),0,1), 
                            columns = ['amplitude','energy', 'iteration', 'macroatom_id'])
    target = DataFrame(np.swapaxes(np.vstack([np.array(atoms_df2['amplitude'])*signs_t,np.array(atoms_df2['energy']),np.array(atoms_df2['iteration']),np.array(atoms_df2['macroatom_id']) ]),0,1), 
                            columns = ['amplitude','energy', 'iteration', 'macroatom_id'])

    atoms_amplitude=[]
    atoms_amplitude2=[]

    atom_energies =[]
    atom_energies2=[]
    fn = partial(get_atoms_amplitude_for_test,mp_params=[channel_names,fs], ref_channel='average')

    for i,picked_atom in enumerate(np.arange(macroatom_reconstrucions.shape[0])):
    # for i,picked_atom in enumerate(chosen_iterations[:3]):
            picked_atoms = nontarget.loc[nontarget['iteration']==picked_atom]
            atoms_amplitude_signs = process_map(fn, picked_atoms.groupby(by='macroatom_id'),desc='Extracting {} atoms {} amplitudes'.format(labels[1],picked_atom),max_workers =6, chunksize=1)

            atoms_amplitude_signs = np.array([np.swapaxes(i.reshape((i.shape[0],i.shape[2])),0,1) for i in atoms_amplitude_signs])

            atoms_amplitude.append([i[0] for i in atoms_amplitude_signs])
            atom_energies.append([i[1] for i in atoms_amplitude_signs])


            picked_atoms = target.loc[target['iteration']==picked_atom]
            atoms_amplitude_signs= process_map(fn, picked_atoms.groupby(by='macroatom_id'),desc='Extracting {} atoms {} amplitudes'.format(labels[0],picked_atom),max_workers =6, chunksize=1)      
                  
            atoms_amplitude_signs = np.array([np.swapaxes(i.reshape((i.shape[0],i.shape[2])),0,1) for i in atoms_amplitude_signs])

            atoms_amplitude2.append([i[0] for i in atoms_amplitude_signs])
            atom_energies2.append([i[1] for i in atoms_amplitude_signs])
    atoms_amplitude=np.array(atoms_amplitude)
    atoms_amplitude2=np.array(atoms_amplitude2)

    atom_energies=np.swapaxes(np.array(atom_energies),0,2)
    atom_energies2=np.swapaxes(np.array(atom_energies2),0,2)

    shape =atom_energies.shape
    shape2=atom_energies2.shape
    stata_wybrane=[]

    for i,x in enumerate(np.arange(atoms_amplitude.shape[0])):
        stata_z_kanalu=[]   
        for j,ch in enumerate(channel_names):
            if ch in p300_channels:
                # x=atoms_amplitude[i,:,j]
                # y=atoms_amplitude2[i,:,j] 
                x=atom_energies[j,:,i]
                y=atom_energies2[j,:,i]           
                # res= stats.permutation_test((x,y),statistic,vectorized=True,alternative='less')
                # st =1-res.pvalue         
                # print(np.mean(sig),np.mean(sig2),i)    
                # st, pv = stats.ttest_ind(sig2,sig)
                st=statistic(y,x,0)
                stata_z_kanalu.append((i,st))
        if i in chosen_iterations:
            stata_wybrane.append(sorted(stata_z_kanalu,key=lambda x:x[1])[-1])
    print('stata_wybrane: ',len(stata_wybrane))
    stata_wybrane=list(reversed(sorted(stata_wybrane,key=lambda x:x[1])))
    chosen_atoms = [i[0] for i in stata_wybrane]
    # chosen_atoms.remove(0)
    print('chosen_atoms: ',chosen_atoms)

    chosen_atoms_freqs=[list(atoms_df2.loc[atoms_df2['iteration']==i]['frequency'])[0] for i in chosen_atoms]
    trend_liniowy_list = [chosen_atoms[chosen_atoms_freqs.index(i)] for i in sorted(chosen_atoms_freqs) if i<1]


    chosen_atoms=[i for i in chosen_atoms if i not in trend_liniowy_list[1:]][:4]



    ##########this part is rather not important for anything other than returns of this function#######################
    print('chosen_atoms: ',chosen_atoms)

    M = sorted(chosen_atoms)[0]
    try:
        caf=0
        while True:
            if chosen_atoms_freqs[caf] <1:
                caf+=1
            else:
                M1 = chosen_atoms[caf]
                break
    except:
        M1 = chosen_atoms[0]


    trend_liniowy=trend_liniowy_list[0]
    # M1=trend_liniowy
    print('trend liniowy: ',trend_liniowy,'\nfreqs: ',sorted(chosen_atoms_freqs))
    print(M1 , chosen_atoms_freqs,chosen_atoms)



    Z = [i[1] for i in stata_wybrane if i[0]==M][0]
    Z1 = [i[1] for i in stata_wybrane if i[0]==M1][0]
    print('M_1: ',M,M1)
################################################################################

################test on the sum of Gabor energies from iterations in chosen_atoms#########################
    epoch_data = [np.sum(np.array([atom_energies[:,:,i] for i in chosen_atoms]),axis=0),
                    np.sum(np.array([atom_energies2[:,:,i] for i in chosen_atoms]),axis=0)]
    print(atom_energies.shape, atom_energies2.shape)
    p_value=[]
    st_value = []
    for i in range(len(channel_names)):
        ch = channel_names[i]
        if ch in p300_channels:
            x=epoch_data[0][i]
            y=epoch_data[1][i]
            #less bacause the order is x=ntgt , y=tgt and we expect the alternative hypothesis, thus low p_value        
            res= stats.permutation_test((x,y),statistic,vectorized=True,alternative="less")
            st=statistic(y,x,0)
            p_value.append(res.pvalue)
            st_value.append(st)

    #holm-bonferroni correction of p_values     
    p_adjusted = multipletests(p_value, alpha=0.05, method='holm')

    pvalue=sorted(p_adjusted[1])[0] 

    Z=sorted(st_value)[-1]   

    # pvalue=0
    # p_adjusted=0


    #############################cluster test on sum of Gabor funtion in chosen_atoms##########################################
    # epoch_data = [np.sum(atom_energies.reshape((shape[0],shape[1],shape[2],1)),axis=2),
    #                 np.sum(atom_energies2.reshape((shape2[0],shape2[1],shape2[2],1)),axis=2)]
    # roi =[None,None]
    # clusters_per_chnl= do_permutation_test(epoch_data,roi,channel_names)

    # possible =[]

    # for i in range(len(channel_names)):

    #     cl, p_val = clusters_per_chnl[i]


    #     for cc, pp in zip(cl, p_val):
    #         possible.append(((cc[0].start,cc[0].stop),pp))
    # atom_range = (len(possible),len(possible))
    # possible=sorted(possible,key=lambda x:x[1])[0]
    # p_value = possible[1]
    # atom_range = possible[0]#klaster 
    # p_value=0
    # atom_range =[0,0]


    ##########################visualization of calculated data###################
    # fig=pb.figure(figsize=(25,15))
    # params = {#'legend.fontsize': 'x-large',
    #     #'figure.figsize': (15, 5),
    #     'legend.title_fontsize': 24,
    #     'axes.labelsize': 24,
    #     'axes.titlesize':24,
    #     'xtick.labelsize':24,
    #     'ytick.labelsize':24}
    # pb.rcParams.update(params)
    # pb.tight_layout()
    # s=1 
    # ym=[np.sum(evoked[j,:,:],axis=0) for j in range(len(channel_names)) if channel_names[j] in p300_channels]
    # ym2=[np.sum(evoked2[j,:,:],axis=0) for j in range(len(channel_names)) if channel_names[j] in p300_channels]

    # ym3=[np.sum(np.array([evoked[i,j] for j in chosen_atoms]),axis=0) for i in range(len(channel_names)) if channel_names[i] in p300_channels]
    # ym4=[np.sum(np.array([evoked2[i,j] for j in chosen_atoms]),axis=0) for i in range(len(channel_names)) if channel_names[i] in p300_channels]    
    # y_max=np.max([np.max(i) for i in ym+ym2+ym3+ym4])
    # y_min=np.min([np.min(i) for i in ym+ym2+ym3+ym4]   )

    # for i in range(len(channel_names)):

    #     if channel_names[i] in p300_channels:
    #         # pb.subplot(4,len(channel_names)//4+1,s)
    #         pb.subplot(4,len(p300_channels)//4+1,s)
    #         # pb.plot(t,np.sum([evoked[i,ch] for ch in chosen_iterations2],0), alpha =0.5, label=labels[1])
    #         # pb.plot(t,np.sum([evoked2[i,ch] for ch in chosen_iterations2],0), alpha =0.5, label=labels[0])
    #         pb.plot(t,np.sum(evoked[i,:,:],axis=0), alpha =0.3, label=labels[1],lw=4)
    #         pb.plot(t,np.sum(evoked2[i,:,:],axis=0), alpha =0.3, label=labels[0],lw=4)           
    #         # pb.plot(t,evoked2[i,j],label=labels[0]+'_{}'.format(j), linestyle='dashed')

    #         # pb.plot(t,evoked[i,j],label=labels[1]+'_{}'.format(j), linestyle='dashed')

    #         pb.plot(t,np.sum(np.array([evoked2[i,j] for j in chosen_atoms]),axis=0),label=labels[0]+'_{}'.format('uśredniona_suma_wybranych_czterech_iteracji'), linestyle='dashed',lw=5)

    #         pb.plot(t,np.sum(np.array([evoked[i,j] for j in chosen_atoms]),axis=0),label=labels[1]+'_{}'.format('uśredniona_suma_wybranych_czterech_iteracji'), linestyle='dashed',lw=5)
    #         # pb.plot(t,np.sum(np.array([evoked2[i,j],evoked2[i,k]]),0),label=labels[0]+'_{}'.format([j,k]), linestyle='dashed')

    #         # pb.plot(t,np.sum(np.array([evoked[i,j],evoked[i,k]]),0),label=labels[1]+'_{}'.format([j,k]), linestyle='dashed')
    #         pb.ylim(y_min-0.05,y_max+0.05)
    #         pb.title(channel_names[i])
    #         s+=1
    # pb.tight_layout()
    # fig.subplots_adjust(top=0.88)
    # pb.suptitle('{}\nUśredniona po odcinkach suma Gaborów dopasowanych algorytmem MP\n(rekonstrukcja rzeczywistego ERP)'.format(name),fontsize=26)
    # # leg = pb.legend(fontsize="24",framealpha=1)
    # # for line in leg.get_lines():
    # #     line.set_linewidth(5.0)
    # pb.savefig(os.path.join(outdir+'/obrazki','{}_{}.png'.format(name,row)))
    # pb.close()
    return pvalue,[0,0],Z,p_adjusted,M,M1,[[i[0] for i in stata_wybrane],chosen_atoms_freqs],Z,Z1

if __name__ == '__main__':
    data = pd.read_csv("/mnt/c/Users/Piotr/Desktop/pliki_pulpit/budzik/p300-wzrokowe/Measurements_database_merged.csv")
    data = data.loc[data['measurement_type']=='p300-wzrokowe']
    names=np.array(data[['person_id','diag']])

    clear = lambda: os.system('clear')

    #outdir is only used to save the *.png and *.txt files; add custome or use sys.argv
    outdir = "/mnt/c/Users/Piotr/Desktop/obrazki_erp"
    try:
        os.makedirs(outdir) 
    except OSError:
        pass 


    rows = [i[0] for i in names]
    for path, subdirs, files in os.walk(sys.argv[1]):
        number_of_db_files_in_path = len([i for i in files if '.db' in i[-3:]])
        for db_name in files:  
            if '.db' in db_name[-3:] and '/mp_books/' in path :#and number_of_db_files_in_path==1:
                paths_to_db = os.path.join(path,db_name)

                temp = os.path.dirname(os.path.dirname(path)).split('/')[-1]
                row =rows.index(temp)
                rows[row]='X'
                if row>=0:
                # if row in [4,38,5,30,53,31,8,24,2,17,19,43,46,34,45,37]:                    
                    # if  row in [0,1,2]:#,38,53,7,45,46,34]:
                    try:
                        print('Running calculations on {}'.format(paths_to_db))
                        diag = names[row][1]
                        p_value,atom_range,p_value2,p_value3,M,M1,chosen_atoms,Z,Z1 = main(paths_to_db,row,names[row][0],outdir,diag)

                        print(names[row][0], names[row][1])

                        with open(os.path.join(outdir, 'p_values_z_sumy_energi_z_wybranych_iteracji_po_korekcie_holm.txt'),'a') as f:            
                            f.write('{} {} {} {} {}\n'.format(names[row][0], names[row][1], p_value, atom_range[0], atom_range[1]))  
                            f.close()    
  
                        with open(os.path.join(outdir, 'statystyka_Z_z_sumy_energi_z_wybranych_iteracji.txt'),'a') as f:            
                            f.write('{} {} {} {} {}\n'.format(names[row][0], names[row][1], Z, atom_range[0], atom_range[1]))   
                            f.close()  

                        with open(os.path.join(outdir, 'chosen_atoms_ze_statystyki_po_energiach.txt'),'a') as f:            
                            f.write('{} {} {} {} {}\n'.format(names[row][0], names[row][1], chosen_atoms, M,M1))  
                            f.close()        
                              
                        with open(os.path.join(outdir, 'p_adjusted_po_energiach.txt'),'a') as f:            
                            f.write('{}\n'.format(p_value3))  
                            f.close()    
                    except Exception as e:
                        print(e)

                        with open(os.path.join(outdir, 'p_values_z_sumy_energi_z_wybranych_iteracji_po_korekcie_holm.txt'),'a') as f:            
                            f.write('{} {} {} {} {}\n'.format(names[row][0], names[row][1], 1, np.random.randint(0,15), np.random.randint(0,15)))  
                            f.close()
                        with open(os.path.join(outdir, 'statystyka_Z_z_sumy_energi_z_wybranych_iteracji.txt'),'a') as f:            
                            f.write('{} {} {} {} {}\n'.format(names[row][0], names[row][1], 1, np.random.randint(0,15), np.random.randint(0,15)))  
                            f.close()                       
                        with open(os.path.join(outdir, 'chosen_atoms_ze_statystyki_po_energiach.txt'),'a') as f:            
                            f.write('{}\n'.format('[]'))  
                            f.close() 
                        with open(os.path.join(outdir, 'p_adjusted_po_energiach.txt'),'a') as f:            
                            f.write('{}\n'.format([]))  
                            f.close()                                 
                        pass