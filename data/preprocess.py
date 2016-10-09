import os
import sys
import argparse
import numpy as np
from scipy.ndimage import imread
import h5py
import pdb

n_ex_per_class = 20
im_size = 28

''' 
TODO 
- downsize images to 28x28
- augment data with rotations
'''

'''
From Brendan Lake's demo code
'''
def load_image(f):
    im = imread(f, flatten=True)
    im = np.logical_not(np.array(im, dtype=bool))
    return im.astype(float)

def rotate_image():

    return

'''
Data returned is in format

   [ class 1, ex 1]
   [ class 1, ex 2]
        ...
   [ class 1, ex20]
   [ class 2, ex 1]
        ...
   [ class N, ex20]

'''
def load_data(path):
    alphabets = os.listdir(path)
    n_classes = 0
    for alphabet in alphabets:
        n_classes += len(os.listdir(path + '/' + alphabet))
    data = np.zeros((n_classes*n_ex_per_class, im_size, im_size))
    
    for alphabet in alphabets:
        alpha_path = path + '/' + alphabet
        for char in os.listdir(alpha_path):
            char_path = alpha_path + '/' + char
            char_idx = int(os.listdir(char_path)[0].split('.')[0].split('_')[0]) - 1 # get char index
            char_offset = char_idx * n_ex_per_class
            for im in os.listdir(char_path):
                # for each character, get the numbering and place in the appropriate spot
                ex_idx = int(im.split('.')[0].split('_')[1])
                data[char_offset+ex_idx-1,:,:] = \
                        load_image(char_path + '/' + im)
    return data, n_classes

'''
Data will be arranged as follows

[ episode 1 ] -->   [ S ] total ((k + kb), im_size, im_size)
[ episode 2 ]       [ B ]   or ((k + kb), 1) for labels
    ...
[ episode N ]

'''
def create_episodes(args, data):
    k, kB = args.k, args.kB
    n_classes = data.shape[0] / n_ex_per_class
    n_examples = k + kB
    base_bat_offset = args.N * k
    inputs = np.zeros((args.n_episodes, n_examples, im_size, im_size))
    outputs = np.zeros((args.n_episodes, n_examples, 1))

    # pre-sample the N classes for each episode
    # maybe too expensive to keep in memory?
    episode_classes = np.random.random_integers(0, n_classes-1, (args.n_episodes, args.N))
    for i,classes in enumerate(episode_classes):
        for j,c in enumerate(classes):
            exs = np.random.choice(n_ex_per_class, n_examples) + (c * n_ex_per_class) # offsets
            set_offset = j*k
            bat_offset = base_bat_offset + j*kB
            inputs[i,set_offset:set_offset+k,:,:] = data[exs[:k],:,:]
            inputs[i,bat_offset:bat_offset+kB,:,:] = \
                    data[exs[k:],:,:]
            outputs[i,set_offset:sef_offset+k,:] = np.ones(k)*c
            outputs[i,bat_offset:bat_offset+kB,:] = np.ones(kB)*c

    return inputs, outputs

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_path', help='path to Omniglot data', type=str, default='')
    parser.add_argument('--outfile', help='path to write data to', type=str, default='omniglot.hdf5')
    parser.add_argument('--N', help='number of unique classes per episode', type=int, default=5)
    parser.add_argument('--k', help='number of examples per class in the support set', type=int, default=1)
    parser.add_argument('--B', help='number of batch examples', type=int, default=10)
    parser.add_argument('--kB', help='number of examples per class in the batch', type=int, default=2) # note: this may not be what they do in paper
    parser.add_argument('--n_episodes', help='number of episodes', type=int, default=10000)
    parser.add_argument('--n_test_classes', help='number of classes saved for test', type=float, default=423)
    args = parser.parse_args(arguments)

    # load all the examples for every class
    print 'Loading data...'
    tr_data, n_tr_classes  = load_data(args.data_path + '/' + 'images_background')
    te_data, n_te_classes = load_data(args.data_path + '/' + 'images_evaluation')
    print '\tData loaded!'

    # augment data
    #augment()

    # create training episodes
    print 'Creating data...'
    tr_ins, tr_outs = create_episodes(args, tr_data)
    te_ins, te_outs = create_episodes(args, te_data)
    print '\tData created!'
    pdb.set_trace()

    # write to hdf5
    with hypy.File(args.outfile, 'w') as f:
        f['tr_ins'] = tr_ins
        f['tr_outs'] = tr_outs
        f['te_ins'] = te_ins
        f['te_outs'] = te_outs
        f['N'] = np.array([args.N], dtype=np.int32)
        f['k'] = np.array([args.k], dtype=np.int32)
        f['kB'] = np.array([args.kB], dtype=np.int32)
    print 'Data written to %s' % args.outfile

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
