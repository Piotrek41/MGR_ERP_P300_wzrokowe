import os

import mne
from multiprocessing.pool import Pool

import joblib
import tqdm
from mne.channels.layout import _find_topomap_coords
from mne.io.pick import _pick_data_channels, pick_info
from mne.viz.topomap import _GridData
import numpy as np

# from coma_structures.utils.utils import convolved_features_vmax, convolved_features_vmin

import collections
import functools

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.abc.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)

# there shouldn't be repeated atoms anymore
# @memoized
def _interpolate_single_feature(interpolator, feature, n_feature, n_channel):
    interpolator.set_values(feature)
    Zi = interpolator()
    return Zi

@memoized
def get_interpolator(pos):
    pos = np.array(pos).reshape(-1, 2)
    interpolator = _GridData(pos, 'local', None)
    return interpolator

INTERP_SIZE = 16

@memoized
def get_positions(channels):
    montage = mne.channels.make_standard_montage('standard_1005')
    sfreq = 128  # doesnt matter anyway
    if len(channels[0].split('-')) == 2:
        channels_source = [i.split('-')[0] for i in channels]
        channels_ref = [i.split('-')[1] for i in channels]
        channels_source_unique = list(set(channels_source))
        channels_ref_unique = list(set(channels_ref))

        pos_source = mne.create_info(channels_source_unique, sfreq, ch_types='eeg')
        pos_reference = mne.create_info(channels_ref_unique, sfreq, ch_types='eeg')
        pos_source.set_montage(montage)
        pos_reference.set_montage(montage)

        picks_source = _pick_data_channels(pos_source)  # pick only data channels
        picks_ref = _pick_data_channels(pos_reference)  # pick only data channels
        pos_source = pick_info(pos_source, picks_source)
        pos_reference = pick_info(pos_reference, picks_ref)
        pos_source = _find_topomap_coords(pos_source, picks=picks_source)
        pos_reference = _find_topomap_coords(pos_reference, picks=picks_ref)

        pos_in_between = []
        for channel_source, channel_ref in zip(channels_source, channels_ref):
            pos_in_between.append((pos_source[channels_source_unique.index(channel_source)] + pos_reference[channels_ref_unique.index(channel_ref)]) / 2)
        pos_in_between = np.array(pos_in_between)
        return pos_in_between

    else:
        pos = mne.create_info(channels, sfreq, ch_types='eeg')
        pos.set_montage(montage)
        picks = _pick_data_channels(pos)  # pick only data channels
        pos = pick_info(pos, picks)
        pos = _find_topomap_coords(pos, picks=picks)
        return pos

def interpolate_feature(data, pos):
    '''data - numpy array with: (N_examples, N_atoms, channels, features),
    can have multiple atoms per example'''
    # these values are selected to maximize usage of available pixels and minimise usage of memory
    res = INTERP_SIZE  # 16 is almost too much for memory usage...
    xmin = -1.5
    xmax = 1.5
    ymin = -1.4
    ymax = 1.6

    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)
    n_examples = data.shape[0]
    n_atoms = data.shape[1]
    n_features = data.shape[3]
    interpolator = get_interpolator(tuple(pos.flatten()))
    interpolator.set_locations(Xi, Yi)
    datas = np.zeros((n_examples, n_atoms, res, res, n_features))


    # for n_ex in tqdm.tqdm(range(n_examples), total=n_examples):
    for n_ex in range(n_examples):
        for n_atom in range(n_atoms):
            for n_feature in range(n_features):
                feature = data[n_ex, n_atom, :, n_feature]
                datas[n_ex, n_atom, :, :, n_feature] = _interpolate_single_feature(interpolator, tuple(feature), n_feature, len(pos))
            # if np.max((data[n_ex, n_atom, :, 0])) > 0.40:
            #     pb.figure()
            #     pb.imshow((datas[n_ex, n_atom, :, :, 0]), vmin=0, vmax=255)
            #     pb.figure()
            #     pb.imshow((datas[n_ex, n_atom, :, :, 1]), vmin=0, vmax=255)
            #     pb.show()
    return datas

