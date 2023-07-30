import os
import numpy as np
from astropy.io.fits.util import path_like
from mne.utils import _ensure_int, logger
import os.path as op

_multi = {
    'str': (str,),
    # 'numeric': (np.floating, float, int_like),
    'path-like': path_like,
    # 'int-like': (int_like,),
    # 'callable': (_Callable(),),
}


def read_freesurfer_lut(fname=None):
    try:
        """Read a Freesurfer-formatted LUT.
        Parameters
        ----------
        fname : str | None
            The filename. Can be None to read the standard Freesurfer LUT.
        Returns
        -------
        atlas_ids : dict
            Mapping from label names to IDs.
        colors : dict
            Mapping from label names to colors.
        """
        lut = _get_lut(fname)
        names, ids = lut['name'], lut['id']
        colors = np.array([lut['R'], lut['G'], lut['B'], lut['A']], float).T
        atlas_ids = dict(zip(names, ids))
        colors = dict(zip(names, colors))
    except Exception as e: # assume unsupported mne
        print(e)
        return [], []
    return atlas_ids, colors



def _validate_type(item, types=None, item_name=None, type_name=None):
    """Validate that `item` is an instance of `types`.
    Parameters
    ----------
    item : object
        The thing to be checked.
    types : type | str | tuple of types | tuple of str
         The types to be checked against.
         If str, must be one of {'int', 'str', 'numeric', 'info', 'path-like',
         'callable'}.
    item_name : str | None
        Name of the item to show inside the error message.
    type_name : str | None
        Possible types to show inside the error message that the checked item
        can be.
    """
    if types == "int":
        _ensure_int(item, name=item_name)
        return  # terminate prematurely
    elif types == "info":
        from mne.io import Info as types

    if not isinstance(types, (list, tuple)):
        types = [types]

    check_types = sum(((type(None),) if type_ is None else (type_,)
                       if not isinstance(type_, str) else _multi[type_]
                       for type_ in types), ())
    if not isinstance(item, check_types):
        if type_name is None:
            type_name = ['None' if cls_ is None else cls_.__name__
                         if not isinstance(cls_, str) else cls_
                         for cls_ in types]
            if len(type_name) == 1:
                type_name = type_name[0]
            elif len(type_name) == 2:
                type_name = ' or '.join(type_name)
            else:
                type_name[-1] = 'or ' + type_name[-1]
                type_name = ', '.join(type_name)
        _item_name = 'Item' if item_name is None else item_name
        raise TypeError("{_item_name} must be an instance of {type_name}, "
                        "got {type(item)} instead")

def _check_fname(fname, overwrite=False, must_exist=False, name='File',
                 need_dir=False):
    """Check for file existence."""
    _validate_type(fname, 'path-like', name)
    if op.exists(fname):
        if not overwrite:
            raise FileExistsError('Destination file exists. Please use option '
                                  '"overwrite=True" to force overwriting.')
        elif overwrite != 'read':
            logger.info('Overwriting existing file.')
        if must_exist:
            if need_dir:
                if not op.isdir(fname):
                    raise IOError(
                        'Need a directory for {name} but found a file '
                        'at {fname}')
            else:
                if not op.isfile(fname):
                    raise IOError(
                        'Need a file for {name} but found a directory '
                        'at {fname}')
            if not os.access(fname, os.R_OK):
                raise PermissionError(
                    '{name} does not have read permissions: {fname}')
    elif must_exist:
        raise FileNotFoundError('{name} does not exist: {fname}')
    return str(op.abspath(fname))


def _get_lut(fname=None):
    """Get a FreeSurfer LUT."""
    _validate_type(fname, ('path-like', None), 'fname')
    if fname is None:
        fname = op.join(op.dirname(__file__), 'data', 'FreeSurferColorLUT.txt')
    _check_fname(fname, 'read', must_exist=True)
    dtype = [('id', '<i8'), ('name', 'U'),
             ('R', '<i8'), ('G', '<i8'), ('B', '<i8'), ('A', '<i8')]
    lut = {d[0]: list() for d in dtype}
    with open(fname, 'r') as fid:
        for line in fid:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            line = line.split()
            if len(line) != len(dtype):
                raise RuntimeError('LUT is improperly formatted: {fname}')
            for d, part in zip(dtype, line):
                lut[d[0]].append(part)
    lut = {d[0]: np.array(lut[d[0]], dtype=d[1]) for d in dtype}
    assert len(lut['name']) > 0
    return lut