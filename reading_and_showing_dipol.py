#!/usr/bin/python
# coding: utf8
import pylab as pb
import os
import numpy as np
import pandas as pd
import mne
import sys
import math
from tqdm.contrib.concurrent import process_map
from mne.datasets import fetch_fsaverage
from brain import get_brain_pictures
from collections import Counter
from cortical_distance import calculate_cortical_distance
dir = fetch_fsaverage()
subjects_dir = os.path.join(dir, '..')
transform_path = os.path.join(dir, 'bem', 'fsaverage-trans.fif')
transform = mne.read_trans(transform_path, return_all=False, )
BRAIN_LENGTH = 200
BRAIN_WIDTH = 180
BRAIN_HEIGHT = 180
default_chnls_to_draw = ('Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4',
                         'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2')
def scatter_3d_summarazie_iterations(X,Y,Z,rgb,chosen_iterations, save_name,title,evoked,t,channel_names,amplitude,frequencies,positions2):

    mni_coords=[]
    iteration=[]
    for it in range(len(X)):
        try:

            x,y,z=X[it],Y[it],Z[it]
            dip_positions =np.swapaxes(np.array([x,y,z]),0,1)        
            mni_dip_pose = mne.head_to_mni(dip_positions, 'fsaverage', transform,
                                        subjects_dir=subjects_dir, verbose=None)
            try:
                mni_x = mni_dip_pose[:, 0]
                mni_y = mni_dip_pose[:, 1]
                mni_z = mni_dip_pose[:, 2]
                
            except IndexError:
                mni_x = np.array([])
                mni_y = np.array([])
                mni_z = np.array([])

            
            if len(list(mni_x))>1:
                mni_coords.append(np.array([mni_x,mni_y,mni_z]))
                iteration.append(chosen_iterations[it])
        except:
            pass

    brain_back, brain_side, brain_top = get_brain_pictures()
    fig = pb.figure(figsize=(20, 15))
    ax = fig.add_subplot(222)

    middle_points=[]#barycenter
    markers_sizes = []#standard deviation of 3d scatters converted into display units
    stds = []
    for i,cords in enumerate(mni_coords):

        middle_point = np.sum(np.array(cords),axis=1)/np.array(cords).shape[1]

        middle_points.append(middle_point)
        std =(np.sum( [(np.array(cords[i])-middle_point[i])**2 for i in range(len(cords))])/len(cords[1]))**0.5
        stds.append(round(std,2))
        r= ax.transData.transform([std,0])[0] - ax.transData.transform([0,0])[0]
        markers_sizes.append(r)
    pb.close()

    middle_points_cords = np.swapaxes(np.array(middle_points),0,1)
    middle_point = np.sum(np.array(middle_points_cords),axis=1)/np.array(middle_points_cords).shape[1]
    dist_from_middle_point =sorted(enumerate([math.dist(middle_point,i) for i in middle_points]),key=lambda x:x[1])[:]
    chosen_iteration = [iteration[u[0]] for u in dist_from_middle_point]
 
    fig = pb.figure(figsize=(20, 15))
    params = {#'legend.fontsize': 'x-large',
            #'figure.figsize': (15, 5),
            'legend.title_fontsize': 20,
            'axes.labelsize': 20,
            'axes.titlesize':20,
            'xtick.labelsize':20,
            'ytick.labelsize':20}
    pb.rcParams.update(params)

    ax = fig.add_subplot(222)
    sc=ax.scatter(positions2[1], positions2[2],  s=amplitude, alpha=0.4, c=frequencies,cmap='rainbow')
    cbr = pb.colorbar(sc,ticks=iteration)
    cbr.set_label('Numer iteracji')
    # cbr.ax.tick_params(labelsize=20)
    for i,positions in enumerate(middle_points):
        color = [j[0] for j in rgb if j[1]==iteration[i]][0]
        # ax.scatter(positions[1], positions[2],  s=markers_sizes[i], alpha=0.2,color=color)
        sc = ax.scatter(positions[1], positions[2],marker='X',edgecolors='black',s=300,  label = '{}_{}'.format(iteration[i],stds[i]),color=color)
        ax.annotate(iteration[i],(positions[1], positions[2]))      

    ax.set_xlabel('y')
    ax.set_ylabel('z')
    pb.title("Sagittal Plane")
    pb.xlim([-BRAIN_LENGTH / 2, BRAIN_LENGTH / 2])
    pb.ylim([-BRAIN_HEIGHT / 2, BRAIN_HEIGHT / 2])
    ax.imshow(brain_side, extent=[-180 / 2 - 19, 180 / 2 - 19, -120 / 2 + 11, 120 / 2 + 11], alpha=0.5)

    ax = fig.add_subplot(221)
    ax.scatter(positions2[0], positions2[2],  s=amplitude, alpha=0.4, c=frequencies,cmap='rainbow')

    for i,positions in enumerate(middle_points):
        color = [j[0] for j in rgb if j[1]==iteration[i]][0]
        # ax.scatter(positions[0], positions[2],  s=markers_sizes[i], alpha=0.2,color=color)
        ax.scatter(positions[0], positions[2],marker='X',edgecolors='black',s=300, label = '{}_{}'.format(iteration[i],stds[i]),color=color)
        ax.annotate(iteration[i],(positions[0], positions[2]))      


    ax.set_xlabel('x')
    ax.set_ylabel('z')

    pb.title('Coronal Plane')
    pb.xlim([-BRAIN_WIDTH / 2, BRAIN_WIDTH / 2])
    pb.ylim([-BRAIN_HEIGHT / 2, BRAIN_HEIGHT / 2])

    ax.imshow(brain_back, extent=[-140 / 2 + 1, 140 / 2 + 1, -100 / 2 + 18, 100 / 2 + 18], alpha=0.5)

    ax = fig.add_subplot(223)
    ax.scatter(positions2[0], positions2[1],  s=amplitude, alpha=0.4, c=frequencies,cmap='rainbow')
    for i,positions in enumerate(middle_points):
        color = [j[0] for j in rgb if j[1]==iteration[i]][0]
        # ax.scatter(positions[0], positions[1],  s=markers_sizes[i], alpha=0.2,color=color)
        ax.scatter(positions[0], positions[1],marker='X',edgecolors='black',s=300,  label = '{}'.format(iteration[i]),color=color)
        ax.annotate(iteration[i],(positions[0], positions[1]))      

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    pb.title('Axial Plane')
    ax.imshow(brain_top, extent=[-180 / 2 + 1, 180 / 2 + 1, -180 / 2 + 11 - 20, 180 / 2 + 11 - 20], alpha=0.5)

    pb.xlim([-BRAIN_WIDTH / 2, BRAIN_WIDTH / 2])
    pb.ylim([-BRAIN_LENGTH / 2, BRAIN_LENGTH / 2])
    # pb.legend(title='Punkt środkowy',fontsize="20",framealpha=1)

    ax = fig.add_subplot(224)
    P3=channel_names.index('P3')
    for ci in chosen_iteration:
        color = [j[0] for j in rgb if j[1]==ci][0]
        sc =ax.plot(t,evoked[P3,ci,:],label=str(ci),color=color)
    sc =ax.plot(t,sum([evoked[P3,i,:] for i in chosen_iteration]),linestyle='dashed',color='black',label='suma')
    # pb.legend(fontsize="20",framealpha=1)
    pb.title('Uśrednione Gabory poszczególnych iteracji na elektrodzie P3')    

    # pb.title('Iterations ordered based on the smallest spread: {}'.format(chosen_iteration))

    pb.tight_layout()
    fig.subplots_adjust(top=0.88)
    # pb.suptitle('Middle points of iterations + standard deviation\n{}'.format(title),fontsize=20)
    pb.suptitle('{}\nAtomy dipolowe {} M Gaborów\nbrakujące kanały: {}\n{}'.format(save_name.split('/')[-1].split('_')[3],
    save_name.split('/')[-1].split('_')[2],
    list(set(default_chnls_to_draw)-set(channel_names)),
    title),fontsize=20)
    # pb.legend()
    fig.savefig(save_name+"_summary_3d_plots.png",dpi=300)
    
    pb.close(fig)    
    return middle_points,middle_point,sum([evoked[P3,i,:] for i in chosen_iteration])               
def gabor(t, s, t0, f0, phase=0.0, segment_length_s=20.0):
    """
    Generates values for Gabor atom with unit amplitude.
    t0 - global, t - also global,
    """

    t_segment = t % segment_length_s
    t0_segment = t0 % segment_length_s
    result = np.exp(-np.pi*((t_segment-t0_segment)/s)**2) * np.cos(2*np.pi*f0*(t_segment-t0_segment) + phase)
    return result
def scatter_3d(positions, frequencies, amplitude, save_name,atoms,t,iteration,it_,channel_names,evoked,histogram_fig,subplot,M_atoms,title='',dpi=None,draw_3d=True):
    brain_back, brain_side, brain_top = get_brain_pictures()

    ax = histogram_fig.add_subplot(subplot)
    histogram = []
    for i in it_:
        histogram.extend([i[0],]*i[1])
    pb.title('Histogram liczby bąbelków {}\n{}'.format(save_name.split('/')[-1].split('_')[2],M_atoms))
    n, bins, patches = ax.hist(histogram,bins=np.arange(-0.5,15),edgecolor='black',rwidth=0.6)
  
    ax.set_xticks(ticks=np.arange(0, 15)) 
    ax.margins(x=0.02) 
    for idx, value in enumerate(n):
        if value > 0:
            pb.text(np.arange(0,15)[idx], value+1, int(value), ha='center')

    fig = pb.figure(figsize=(20, 22.5))
    ax = fig.add_subplot(222)
    sc = ax.scatter(positions[1], positions[2],  s=amplitude, alpha=0.4, c=frequencies,cmap='rainbow')
    cbr = pb.colorbar(sc,ticks=iteration)
    cbr.set_label('Numer iteracji')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    rgb=sc.to_rgba(frequencies)
    rgb=list(set(zip([tuple(i) for i in rgb],frequencies)))
    pb.title("Sagittal Plane")
    pb.xlim([-BRAIN_LENGTH / 2, BRAIN_LENGTH / 2])
    pb.ylim([-BRAIN_HEIGHT / 2, BRAIN_HEIGHT / 2])
    ax.imshow(brain_side, extent=[-180 / 2 - 19, 180 / 2 - 19, -120 / 2 + 11, 120 / 2 + 11], alpha=0.5)    

    pb.close(fig)
    return rgb
def groups(group):
    example = group[1]
    return example
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

            if atom['frequency']<7 and 0.2<atom['absolute_position']<0.95:
                temp.append(1) 

            sig = gabor(t, atom['width'],atom['absolute_position'], atom['frequency'], atom['phase'], segment_length)
            sig*= abs(atom['amplitude'])

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
def main(pickle_file, M_atoms, gof,hitogram_fig,subplot):

    file = pd.read_pickle(pickle_file)
    channel_names=list(set(list(file['ch_name'])))
    channel_names =[i for i in default_chnls_to_draw if i in channel_names]
    columns = list(file.columns)

    atoms_df=[]
    for i in channel_names:
        atoms_df.extend(list(np.array(file.loc[file['ch_name']==i].sort_values(['macroatom_id','iteration']))))
    atoms_df=pd.DataFrame(atoms_df,columns=columns)

    try:
        atoms_=atoms_df
        atoms_filered=atoms_.loc[ (atoms_['dip_distance_to_cortex_voxel'] < 50) & (-5<atoms_['dip_distance_to_cortex_voxel']) ]
    except:
        atoms_ = calculate_cortical_distance(atoms_df)
        atoms_.to_pickle(pickle_file)
        atoms_filered=atoms_.loc[ (atoms_['dip_distance_to_cortex_voxel'] < 50) & (-5<atoms_['dip_distance_to_cortex_voxel']) ]



    try:
        fs = np.array(file['fs'])[0]
        segment_length =np.array(file['segment_length'])[0]
    except:
        fs = 256
        segment_length=1.19921875
    dip_gof = sorted(list(Counter([int(i) for i in np.array(file['dip_gof'])]).most_common()), key=lambda x:x[0])[-3][0]
    dip_gof=int(np.mean(np.array(file['dip_gof']),axis=0))
    dip_gof=gof
    macroatom_reconstrucions, evoked, channel_names, fs, segment_length, chosen_iterations =get_macroatom_reconstrucions(atoms_df,channel_names,fs,segment_length,'{}'.format(os.path.basename(pickle_file)))
    # t=np.linspace(0,segment_length,int(fs*segment_length))-0.2
    # pb.plot(t,np.sum(evoked[channel_names.index('P3'),:,:],axis=0))
    # pb.show()

    file = atoms_filered
    x,y,z, freq, amplitude, it = [], [], [], [], [], []
    X,Y,Z = [],[],[]
    # for iteration in sorted(list(set(list(file['iteration'])))):
    for iteration in sorted(M_atoms):
        x_it,y_it,z_it = [], [], []
        mean_amp = file.loc[file['iteration']==iteration]
        mean_amp = np.mean([np.max(np.abs(np.array(mean_amp.loc[mean_amp['macroatom_id'] == m_id]['modulus']))) for m_id in np.array(mean_amp['macroatom_id'])])
        # print(mean_amp)

        for macroatom_id_dipole in list(set(list(file['macroatom_id_dipole']))):
            picked_atoms = file.loc[file['iteration']==iteration]
            picked_atoms = picked_atoms.loc[picked_atoms['macroatom_id_dipole']==macroatom_id_dipole]
            try:
                dip_posx=list(picked_atoms['dip_posx'])[0]
                dip_posy=list(picked_atoms['dip_posy'])[0]
                dip_posz=list(picked_atoms['dip_posz'])[0]

                if np.array( picked_atoms['dip_gof'])[0]>=dip_gof:           
                    x_it.append(dip_posx)
                    y_it.append(dip_posy)
                    z_it.append(dip_posz)
                    freq.append(list(picked_atoms['iteration'])[0])#the variable name is freq but the intent is to select list of iterations
                    # freq.append(np.max(np.array(picked_atoms['modulus'])))#here is an example of selecting real frequency from the database           
                    amplitude.append(abs(np.max(np.array(picked_atoms['amplitude'])))*10)#this parameter is only meant to scale the size of the markers on the visualization plot  
                    
                    it.append(iteration)        
            except:
                pass
        x.extend(x_it)
        y.extend(y_it)
        z.extend(z_it)

        X.append(x_it)
        Y.append(y_it)
        Z.append(z_it)

    print(sorted(enumerate(freq), key = lambda x:x[1])[-1])
    dip_positions =np.swapaxes(np.array([x,y,z]),0,1)        
    mni_dip_pose = mne.head_to_mni(dip_positions, 'fsaverage', transform,
                                subjects_dir=subjects_dir, verbose=None)
    try:
        mni_x = mni_dip_pose[:, 0]
        mni_y = mni_dip_pose[:, 1]
        mni_z = mni_dip_pose[:, 2]
            
    except IndexError:
        mni_x = np.array([])
        mni_y = np.array([])
        mni_z = np.array([])

    positions =np.array([mni_x,mni_y,mni_z])


    if len(M_atoms)>4:
        save_name = pickle_file[:-4]+'_ALL'
    else:
        save_name = pickle_file[:-4]



    it_=list(Counter(it).most_common()) 
    it=sorted(list(set(it)) )    

    t=np.linspace(0,segment_length,int(fs*segment_length))-0.2

    title='dip_gof>{}'.format(dip_gof)
    rgb=scatter_3d(positions, freq, amplitude, save_name+'_{}.png'.format(dip_gof),[evoked[:,i,:] for i in it],t,it,it_,channel_names,evoked,hitogram_fig,subplot,sorted(M_atoms),title=title)
    
    middle_points,middle_point,P3=scatter_3d_summarazie_iterations(X,Y,Z,rgb ,sorted(M_atoms),save_name,title,evoked,t,channel_names,amplitude,freq,positions)
    return middle_points,middle_point,P3
if __name__ == '__main__':

    nb=0
    gof=60

    #this text file contains chosen iterations that will be drawn and the names of the subjects
    # example - M_atom_list=[('M', [1,11,9,3]),('MDv', [1,4,2,7])]

    # with open(sys.argv[1]) as f:
    # with open("/mnt/c/Users/Piotr/Desktop/dips_new/ladne_erpy/ladne_erp_wybrane_4_M_atomow.txt") as f:

    #     M_atom_list = [i.strip('\n').split(' ') for j,i in enumerate(f.readlines())]
    # M_atom_list = [(i[0],ast.literal_eval(i[1])) for i in M_atom_list]


    ERPs=[]
    segment = sys.argv[2]
    for path, subdirs, files in os.walk(sys.argv[1]):
        number_of_db_files_in_path = len([f for f in files if 'pkl' in f[-3:]])
        hitogram_fig =pb.figure(figsize=(20,15))
        subplot=221

        if segment == 'target':
            temp_list=[j for j in files if 'pkl' in j[-3:] and 'nontarget' not in j]
        else: 
            temp_list=[j for j in files if 'pkl' in j[-3:] and 'nontarget' in j]


        for pkl_name in temp_list:  

            # try:
            #     number_in_a_pkl = [num for num in re.findall(r'\d+', pkl_name)]
            #     chosen_atoms_for_the_pkl = [i for i in M_atom_list if i[0] in number_in_a_pkl[0] and len(number_in_a_pkl)>1][0][1]
            # except:
                
            #     chosen_atoms_for_the_pkl = [i for i in M_atom_list if '_{}_'.format(i[0]) in pkl_name][0][1]


            # M_atoms = list(set(list(np.arange(15)))-set(chosen_atoms_for_the_pkl)    )

            print('Working on: ',pkl_name,'\ngof: ',gof)
        

            try:
                # middle_points,middle_point,P3=main(os.path.join(path,pkl_name), chosen_atoms_for_the_pkl, gof,hitogram_fig,subplot)
                # middle_points,middle_point,P3=main(os.path.join(path,pkl_name), M_atoms, gof,hitogram_fig,subplot)
                subplot+=1                
                temp1,temp2,temp3= main(os.path.join(path,pkl_name), list(np.arange(15)), gof,hitogram_fig,subplot)
                subplot+=1

            except Exception as e:
                print(e)
                pass
 
        try:
            pb.tight_layout()
            hitogram_fig.subplots_adjust(top=0.88)
            # pb.suptitle('Middle points of iterations + standard deviation\n{}'.format(title),fontsize=20)
            pb.suptitle('{}\nRozkład liczby bąbelków target i nontarget wybranych i wszystkich iteracji'.format(pkl_name.split('_')[3],fontsize=20))
            # pb.legend()
            hitogram_fig.savefig(os.path.join(path,pkl_name.split('_')[3]+"_histogram.png"),dpi=300)
            
            pb.close(hitogram_fig)          
        except:
            pb.close(hitogram_fig)               
        if number_of_db_files_in_path>0:

            nb+=1

    