import os
import sys
import argparse
import numpy as np
from scipy.ndimage import imread
import h5py
import pdb

'''
From Brendan Lake's demo code
'''
def load_image(f):
    im = imread(f, flatten=True)
    im = np.logical_not(np.array(im, dtype=bool))
    return im.astype(float)

def rotate_image():

    return

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('', help='', type='', default='')
