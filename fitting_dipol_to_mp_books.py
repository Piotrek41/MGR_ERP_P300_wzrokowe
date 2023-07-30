import os
import numpy as np
import sqlite3
import mne
import sys
import time
import json
import pylab as pb
import pandas as pd
import tempfile
from pandas import DataFrame
from tqdm.contrib.concurrent import process_map
from functools import partial
from astropy.stats import circmean
from mne.datasets import fetch_fsaverage


default_chnls_to_draw = ('Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4',
                         'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2', 'M1', 'M2')
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



        ####path to the right text file need to be adjusted if there is more than one mp book in the same folder####

        # empi_params_files= [book_file for book_file in os.listdir(os.path.dirname(atom_db_path)) if book_file.endswith(".txt")]
        # empi_params_file = [i for i in empi_params_files][0]        
        # print(os.path.basename(atom_db_path)[:-3])
        # print(empi_params_file)

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

    columns = ['iteration', 'modulus', 'amplitude', 'width', 'frequency', 'phase', 'struct_len', 'absolute_position', 'offset',
               'ch_id', 'ch_name','macroatom_id']

    dtypes = {'iteration': int, 'modulus': float, 'amplitude': float,
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
        modulus = atom[2]
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

        chosen.append([iteration, modulus,
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

def amplitude_signs(example, debug=False):
    weights = example.amplitude.values / np.sum(example.amplitude.values) * len(example.amplitude.values)
    dominant_polar_plot_direction = circmean(example.phase.values, weights=weights)
    seperating_plot_direction = (dominant_polar_plot_direction + np.pi / 2) % (np.pi * 2)
    rotated_to_seperating_line = (example.phase - seperating_plot_direction) % (np.pi * 2)
    signs = []
    for phase in rotated_to_seperating_line:
        if 0 < phase <= np.pi:
            signs.append(1)
        else:
            signs.append(-1)
    return np.array(signs)

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

def dipole_fitting_func(group, fs_dir,ref_channel='Oz',mp_params=[]):
    """group - groupby pandas object, gives a tuple (nr, pandas sub dataframe)"""
    channels = mp_params[0]
    fs=mp_params[1]
    if ref_channel == 'average':
        ref_channels = []
    else:
        ref_channels = ref_channel.strip().split(',')
        if len(ref_channels) == 1 and ref_channels[0] == '':
            ref_channels = []

    # missing_channels = list(set(default_chnls_to_draw) - set(channels))
    # channels_with_ref = channels + missing_channels
    missing_channels = []
    channels_with_ref = channels

    try:
        example = group[1]
    except:
        example=group
    # example = example[picked_atom]
    # print(example)

    bem_file = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    trans_file = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
    info = mne.create_info(channels_with_ref, fs, ch_types='eeg', verbose='CRITICAL')
    covariance = mne.make_ad_hoc_cov(info)


    # signs = amplitude_signs(example)
    missing_channels_count = len(missing_channels)
    data = np.array(list(example.amplitude.values ) + [0.0, ] * missing_channels_count)[:, None] * 1e-6
    # data = np.array(list(example.amplitude.values*signs ) + [0.0, ] * missing_channels_count)[:, None] * 1e-6
    # print(data.shape)

    evoked = mne.EvokedArray(data,
                         info,
                         verbose='CRITICAL')

    if len(ref_channels) == 0:
        evoked.set_eeg_reference('average', verbose='CRITICAL')
    else:
        evoked.set_eeg_reference(ref_channels, verbose='CRITICAL')
    evoked.set_montage('standard_1020', verbose='CRITICAL')

    dip, residual = mne.fit_dipole(evoked, covariance, bem_file, trans=trans_file, verbose='CRITICAL', min_dist=0,
                               tol=5e-03)

    macroatom_id = example.macroatom_id.values[0]
    posx = dip.pos[0][0]
    posy = dip.pos[0][1]
    posz = dip.pos[0][2]

    orix = dip.ori[0][0]
    oriy = dip.ori[0][1]
    oriz = dip.ori[0][2]
    gof = dip.gof[0]
    amplitude = dip.amplitude[0] # AmperMeters?
    dip_param = [macroatom_id,posx, posy, posz, orix, oriy, oriz, gof, amplitude]


    ch_num = example.ch_id.shape[0]
    # ch_num =11
    return dip_param * ch_num
def show_example(example, example2,subjects_dir, transform,save_path, chnames, show=True, debug=True):
    dipole_pos = [example['dip_posx'], example['dip_posy'], example['dip_posz']]
    dipole_ori = [example['dip_orix'], example['dip_oriy'], example['dip_oriz']]
    gof = example['dip_gof']

    amplitude = example['dip_amplitude']

    number_of_channels=len(chnames)

    dipole_pos=[np.array([x for j,x in enumerate(i) if (j+1)%number_of_channels==0]) for i in dipole_pos]
    dipole_ori=[np.array([x for j,x in enumerate(i) if (j+1)%number_of_channels==0]) for i in dipole_ori]  
  
    gof=[x for j,x in enumerate(gof) if (j+1)%number_of_channels==0] 

    amplitude=[x for j,x in enumerate(amplitude) if (j+1)%number_of_channels==0]



    dipole_pos2 = [example2['dip_posx'], example2['dip_posy'], example2['dip_posz']]
    dipole_ori2 = [example2['dip_orix'], example2['dip_oriy'], example2['dip_oriz']]
    gof2 = example2['dip_gof']

    amplitude2 = example2['dip_amplitude']

    dipole_pos2=[np.array([x for j,x in enumerate(i) if (j+1)%number_of_channels==0]) for i in dipole_pos2]
    dipole_ori2=[np.array([x for j,x in enumerate(i) if (j+1)%number_of_channels==0]) for i in dipole_ori2]  

    gof2=[x for j,x in enumerate(gof2) if (j+1)%number_of_channels==0] 

    amplitude2=[x for j,x in enumerate(amplitude2) if (j+1)%number_of_channels==0]



    dipole_pos = [np.array(list(dipole_pos[0])+list(dipole_pos2[0])),np.array(list(dipole_pos[1])+list(dipole_pos2[1])),np.array(list(dipole_pos[2])+list(dipole_pos2[2]))]
    dipole_ori = [np.array(list(dipole_ori[0])+list(dipole_ori2[0])),np.array(list(dipole_ori[1])+list(dipole_ori2[1])),np.array(list(dipole_ori[2])+list(dipole_ori2[2]))]
    gof = gof+gof2
    amplitude = amplitude+amplitude2

    dip = mne.Dipole(np.zeros(np.array(dipole_pos).T.shape[0]), np.array(dipole_pos).T, amplitude, np.array(dipole_ori).T, gof,
                     name=None, conf=None, khi2=None, nfree=None, verbose=None)

    
    fig = dip.plot_locations(transform, 'fsaverage', subjects_dir, show=False)
    ax = pb.gca()
    # ax.set_title("From file (with reference)")
    fig.tight_layout()

    fig.savefig(save_path)
    if show:
        pb.show()
    else:
        pb.close()
    return fig

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

            if atom['frequency']<4 and 0.2<atom['absolute_position']<0.95:
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

def main(atom_db_path,row, name, outdir, segment):

    atoms_target,atoms_nontarget,channel_names,fs,segment_length,labels = read_db_atoms_dipol(atom_db_path)

    #nontarget
    nontarget = get_atoms(atoms_nontarget, channel_names, width_coeff=1)   
    macroatom_reconstrucions, evoked, channel_names, fs, segment_length, chosen_iterations =get_macroatom_reconstrucions(nontarget,channel_names,fs,segment_length,labels[1])

    #target
    target = get_atoms(atoms_target, channel_names, width_coeff=1)  
    macroatom_reconstrucions2, evoked2, channel_names2, fs2, segment_length2, chosen_iterations2 = get_macroatom_reconstrucions(target,channel_names,fs,segment_length,labels[0])#target

    signs_nt = sings_from_reconstructs(macroatom_reconstrucions) 
    signs_t = sings_from_reconstructs(macroatom_reconstrucions2)  

    nontarget['amplitude'] = nontarget['amplitude'].multiply(signs_nt, axis=0)
    target['amplitude'] = target['amplitude'].multiply(signs_t, axis=0)



    t=np.linspace(0,segment_length,int(fs*segment_length))-0.2




    #############################fitting dipoles######################################
    while True:
        try:
            dir = fetch_fsaverage()
            break
        except (json.decoder.JSONDecodeError, RuntimeError):
            time.sleep(0.1)

    subjects_dir = os.path.join(dir, '..')
    transform_path = os.path.join(dir, 'bem', 'fsaverage-trans.fif')
    transform = mne.read_trans(transform_path, return_all=False, )    
    
    tmp_dir = tempfile.mkdtemp()

    number_of_all_iterations = np.arange(macroatom_reconstrucions.shape[0])
    # #for i,picked_atom in enumerate(chosen_iterations):
    list_of_dipole_dataframes = []
    for i,picked_atom in enumerate(number_of_all_iterations):

        fn = partial(groups)

        if segment == 'target':
            picked_atoms = target.loc[target['iteration']==picked_atom]
        else:
            picked_atoms = nontarget.loc[nontarget['iteration']==picked_atom]

        atom_groups = process_map(fn, picked_atoms.groupby(by='macroatom_id'),desc='grouping atoms {}'.format(picked_atom),max_workers =3, chunksize=1)


        fn = partial(dipole_fitting_func,mp_params=[channel_names,fs], fs_dir=fetch_fsaverage(verbose='CRITICAL'),ref_channel='average')

        if segment == 'target':
            picked_atoms = target.loc[target['iteration']==picked_atom]
        else:
            picked_atoms = nontarget.loc[nontarget['iteration']==picked_atom]

        dipole_params = process_map(fn, picked_atoms.groupby(by='macroatom_id'),desc='FITTING DIPOLE TO ATOM {}'.format(picked_atom),max_workers =3, chunksize=1)
        print(len(dipole_params),len(dipole_params[0]))

        group_names=list(atom_groups[0].columns)
        dipole_param_names = ['macroatom_id_dipole', 'dip_posx', 'dip_posy', 'dip_posz', 'dip_orix', 'dip_oriy',
                                'dip_oriz', 'dip_gof', 'dip_amplitude']
        nb_epoch = int(np.max(picked_atoms['macroatom_id'])+1)
        nb_parameters = len(dipole_param_names)#because ch_id will be added later to dataframe
        nb_channels = len(dipole_params[0])//nb_parameters
        dipole_params=np.array(dipole_params).reshape((nb_epoch,nb_channels,nb_parameters))

        df = []
        for i in range(nb_epoch):
            for ch in range(nb_channels):
                df.append(list(np.array(atom_groups)[i,ch,:])+list(dipole_params[i,ch,:])+[fs,segment_length])
        example =DataFrame(df,columns=group_names+dipole_param_names+['fs','segment_length']) 
        list_of_dipole_dataframes.append(example)


        dataframe_with_all_fitted_dipoles_from_all_iterations = pd.concat(list_of_dipole_dataframes)
    
        columns = list(dataframe_with_all_fitted_dipoles_from_all_iterations.columns)
        atoms_df=[]
        for i in channel_names:
            atoms_df.extend(list(np.array(dataframe_with_all_fitted_dipoles_from_all_iterations.loc[dataframe_with_all_fitted_dipoles_from_all_iterations['ch_name']==i].sort_values(['macroatom_id','iteration']))))
        atoms_df=pd.DataFrame(atoms_df,columns=columns)

        if segment =='target':
            atoms_df.to_pickle(os.path.join(outdir,'fitted_dipole_{}_{}_{}_all_channels.pkl'.format(labels[0],name,row)))
        else:
            atoms_df.to_pickle(os.path.join(outdir,'fitted_dipole_{}_{}_{}_all_channels.pkl'.format(labels[1],name,row)))

if __name__ == '__main__':


    data = pd.read_csv("/mnt/c/Users/Piotr/Desktop/pliki_pulpit/budzik/p300-wzrokowe/Measurements_database_merged.csv")
    data = data.loc[data['measurement_type']=='p300-wzrokowe']
    names=np.array(data[['person_id','diag']])


    #sys.argv[1] --- path to folder or tree of folders with multiple data

    #keep in mind that you need to change how the text file in "read_db_atoms_dipol" function is selceted
    #based on how many mp data sets are in one folder alone
    #i myself have been using single datasets with one txt file, hence the none complicated selection in "read_db_atoms_dipol"
    #also adjust the loops below based on the directories u want your dipoles to be stored at etc.
    rows = [i[0] for i in names]
    segment = sys.argv[2]
    for path, subdirs, files in os.walk(sys.argv[1]):
        number_of_db_files_in_path = len([i for i in files if '.db' in i[-3:]])
        for db_name in files:  
            # if '.db' in db_name[-3:] and not 'mp_books_no_frontal_electrodes' in path and number_of_db_files_in_path==1:
            if '.db' in db_name[-3:] and not 'mp_books_no_frontal_electrodes' in path:
                paths_to_db = os.path.join(path,db_name)

                name_of_the_subject = os.path.dirname(os.path.dirname(path)).split('/')[-1]

                #rows are the order or the row in the Measurements_database_merged.csv database
                row =[i[0] for i in names].index(name_of_the_subject) 
                row =rows.index(name_of_the_subject)
                rows[row]='X'

                if 11>=row>=10:

                    outdir = os.path.join("/mnt/c/Users/Piotr/Desktop/XD/whatever",names[row][0])
                    try:
                        os.makedirs(outdir) 
                    except OSError:
                        pass    
                    main(paths_to_db,row,names[row][0],outdir,segment)         
                    # pkl_files = []
                    # for root, dirs, files in os.walk(outdir):
                    #     for file in files:
                    #         if file.endswith(".pkl"):
                    #             pkl_files.append( file)

                    # if len([i for i in pkl_files if '_nontarget_' in i])==0:
                    #     print('Running calculations on {}'.format(paths_to_db))

                    #     main(paths_to_db,row,names[row][0],outdir)

