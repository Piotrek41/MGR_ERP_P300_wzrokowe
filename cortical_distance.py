import os

import mne
import nibabel
import numpy as np
from mne.datasets import fetch_fsaverage
from tqdm import tqdm

# from coma_structures.research.show_example_dipole import show_example
from interpolate_head_features import memoized
from mne_freesurf_labels import read_freesurfer_lut
from sklearn.neighbors import KDTree as skKDTree

DEBUG = False

@memoized
def find_cortical_voxels(cortex_voxel_values, parcelation_img_path):
    parcelation_img = nibabel.load(parcelation_img_path)
    cortex_indexes = []
    for cortex_voxel_value in tqdm(cortex_voxel_values, desc='getting cortical voxels'):
        cortex_index = np.argwhere(
            parcelation_img.get_fdata() == cortex_voxel_value)
        cortex_indexes.extend(cortex_index)
    cortex_indexes = np.array(cortex_indexes)
    return cortex_indexes


def euclidean_distance(coords1, coords2):
    return np.sqrt(np.sum((coords1 - coords2)**2, axis=1))


def calculate_cortical_distance(atoms_original_sort, dir=None, subject=None):
    # right now using FSAVERAGE brain
    if dir is None and subject is None:
        dir = fetch_fsaverage()
        subjects_dir = os.path.join(dir, '..')
        parcellation_image_path = os.path.join(subjects_dir, 'fsaverage', 'mri', 'aparc.a2005s+aseg.mgz')
        transform_path = os.path.join(dir, 'bem', 'fsaverage-trans.fif')
        transform = mne.read_trans(transform_path, return_all=False, )
        lut_inv_dict = read_freesurfer_lut()[0]

        label_lut = {v: k for k, v in lut_inv_dict.items()}
        parcelation_img = nibabel.load(parcellation_image_path)

        cortical_keys = [key for key in lut_inv_dict.keys() if key.startswith('ctx-')]
        cortical_keys.append('Left-Cerebral-Cortex')
        cortical_keys.append('Right-Cerebral-Cortex')

        cortex_voxel_values = [lut_inv_dict[key] for key in cortical_keys]
    else:
        raise Exception("only fsaverage supported right now")

    cortex_indexes = find_cortical_voxels(tuple(cortex_voxel_values),
                                          parcellation_image_path)

    voxel_to_mni_tr = parcelation_img.header.get_vox2ras_tkr()
    cortical_mni_positions = mne.transforms.apply_trans(voxel_to_mni_tr, cortex_indexes)

    if DEBUG:
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D

        fig = pyplot.figure()
        ax = Axes3D(fig)

        ax.scatter(cortical_mni_positions[::500,0],
                   cortical_mni_positions[::500,1],
                   cortical_mni_positions[::500,2],)
        pyplot.show()

    cortex_kdtree = skKDTree(cortical_mni_positions)

    positions = atoms_original_sort[['dip_posx',
                                    'dip_posy',
                                    'dip_posz'
                                     ]
    ]
    mni_dip_pose = mne.head_to_mni(positions, 'fsaverage', transform,
                                   subjects_dir=subjects_dir, verbose=None)

    min_distances_to_cortex_voxel = np.empty(mni_dip_pose.shape[0])
    further_or_closer_to_coordinate_center = np.empty(mni_dip_pose.shape[0]) # -1 - outside, +1 inside
    closest_cortical_voxel_mni_pos = np.empty((mni_dip_pose.shape[0], 3))

    indexes_cortical_labels = []

    chunksize = 10000
    for i in tqdm(list(range(0, mni_dip_pose.shape[0], chunksize)), desc="finding closest cortical region"):
        # if DEBUG:
        #     if i%56==0:
        #         from matplotlib import pyplot
        #         from mpl_toolkits.mplot3d import Axes3D

        #         fig = pyplot.figure()
        #         ax = Axes3D(fig)

        #         ax.scatter(cortical_mni_positions[::100, 0],
        #                    cortical_mni_positions[::100, 1],
        #                    cortical_mni_positions[::100, 2],s=3 , alpha=0.5)

        #         ax.scatter([mni_dip_pose[i, 0]],
        #                    [mni_dip_pose[i, 1]],
        #                    [mni_dip_pose[i, 2]], s=5, color='r')
        #         example = atoms_original_sort.iloc[i]
        #         example['atom_importance'] = 1.0
        #         macroatom = atoms_original_sort[atoms_original_sort['macroatom_id'] == example['macroatom_id']]
        #         macroatom['atom_importance'] = 1.0
        #         ch_names = macroatom['ch_name'].values
        #         show_example(example, ch_names, subjects_dir, transform, macroatom, show=False, debug=False)
        #         print(example)
        #         pyplot.show()
        dist, index = cortex_kdtree.query(mni_dip_pose[i:i+chunksize])
        dist = np.squeeze(dist)
        index = np.squeeze(index)
        min_distances_to_cortex_voxel[i:i+chunksize] = dist
        indexes_voxel_coord = cortex_indexes[index]
        indexes_mni_coord = cortical_mni_positions[index]
        closest_cortical_voxel_mni_pos[i:i+chunksize, :] = indexes_mni_coord
        indexes_cortical_labels_local = [label_lut[parcelation_img.get_fdata()[index_local[0], index_local[1], index_local[2]]] for index_local in indexes_voxel_coord]
        indexes_cortical_labels.extend(indexes_cortical_labels_local)
        distance_to_anterior_commissure_dipole = euclidean_distance(mni_dip_pose[i:i + chunksize], np.zeros((mni_dip_pose[i:i + chunksize].shape[0], 3)))
        distance_to_anterior_commissure_cortex_voxel = euclidean_distance(indexes_mni_coord, np.zeros((indexes_mni_coord.shape[0], 3)))
        further_or_closer_to_coordinate_center[i:i+chunksize] = np.sign(distance_to_anterior_commissure_cortex_voxel - distance_to_anterior_commissure_dipole)
    further_or_closer_to_coordinate_center[further_or_closer_to_coordinate_center==0] = 1
    min_distances_to_cortex_voxel = min_distances_to_cortex_voxel * further_or_closer_to_coordinate_center

    atoms_original_sort['dip_distance_to_cortex_voxel'] = min_distances_to_cortex_voxel

    # if DEBUG:
    #     import pylab
    #     pylab.figure()
    #     pylab.hist(min_distances_to_cortex_voxel, bins=100)
    #     pylab.title("all atoms")

    #     pylab.figure()
    #     pylab.hist(min_distances_to_cortex_voxel[atoms_original_sort['dip_gof'] >= 90], bins=100)
    #     pylab.title("dip gof >= 90")
    #     pylab.figure()
    #     pylab.hist(
    #         min_distances_to_cortex_voxel[atoms_original_sort['dip_gof'] < 90],
    #         bins=100)
    #     pylab.title("dip gof < 90")
    #     pylab.show()
    atoms_original_sort['dip_closest_cortex_voxel_mni_x'] = closest_cortical_voxel_mni_pos[:, 0]
    atoms_original_sort['dip_closest_cortex_voxel_mni_y'] = closest_cortical_voxel_mni_pos[:, 1]
    atoms_original_sort['dip_closest_cortex_voxel_mni_z'] = closest_cortical_voxel_mni_pos[:, 2]
    atoms_original_sort[
        'dip_closest_cortex_voxel_label'] = indexes_cortical_labels

    return atoms_original_sort