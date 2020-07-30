# utils.py ---
#
# Filename: utils.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Mon Jan 18 16:52:58 2016 (+0100)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
#
# Copyright (C)
# Visual Computing Group @ University of Victoria
# Computer Vision Lab @ EPFL

# Code:

import warnings
#warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore")
import gzip
import pickle

#with warnings.catch_warnings():
import h5py


def savepklz(data_to_dump, dump_file_full_name, force_run=False):
    ''' Saves a pickle object and gzip it '''

    if not force_run:
        raise RuntimeError("This function should no longer be used!")

    with gzip.open(dump_file_full_name, 'wb') as out_file:
        pickle.dump(data_to_dump, out_file)


def loadpklz(dump_file_full_name, force_run=False):
    ''' Loads a gziped pickle object '''

    if not force_run:
        raise RuntimeError("This function should no longer be used!")

    with gzip.open(dump_file_full_name, 'rb') as in_file:
        dump_data = pickle.load(in_file)

    return dump_data


def saveh5(dict_to_dump, dump_file_full_name):
    ''' Saves a dictionary as h5 file '''

    with h5py.File(dump_file_full_name, 'w') as h5file:
        if isinstance(dict_to_dump, list):
            for i, d in enumerate(dict_to_dump):
                newdict = {'dict' + str(i): d}
                writeh5(newdict, h5file)
        else:
            writeh5(dict_to_dump, h5file)


def writeh5(dict_to_dump, h5node):
    ''' Recursive function to write dictionary to h5 nodes '''

    for _key in dict_to_dump.keys():
        if isinstance(dict_to_dump[_key], dict):
            h5node.create_group(_key)
            cur_grp = h5node[_key]
            writeh5(dict_to_dump[_key], cur_grp)
        else:
            h5node[_key] = dict_to_dump[_key]


def loadh5(dump_file_full_name):
    ''' Loads a h5 file as dictionary '''

    try:
        with h5py.File(dump_file_full_name, 'r') as h5file:
            dict_from_file = readh5(h5file)
    except Exception as e:
        print("Error while loading {}".format(dump_file_full_name))
        raise e

    return dict_from_file


def readh5(h5node):
    ''' Recursive function to read h5 nodes as dictionary '''

    dict_from_file = {}
    for _key in h5node.keys():
        if isinstance(h5node[_key], h5py._hl.group.Group):
            dict_from_file[_key] = readh5(h5node[_key])
        else:
            dict_from_file[_key] = h5node[_key].value

    return dict_from_file


#
# utils.py ends here
