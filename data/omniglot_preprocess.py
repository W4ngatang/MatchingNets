import os
import sys
import argparse
import numpy as np
from scipy.ndimage import imread
import h5py
import pdb

n_ex_per_class = 20
im_size = 28
thresh = 128

'''
Due to resizing, images are no longer purely black and white
Rounding to 0 or 1 based on a threshold seems ok
Threshold arbitrarily chosen as 128
'''
def load_image(f):
    original = np.logical_not(imread(f)/thresh).astype(float)
    #return [np.rot90(original, i) for i in xrange(4)]
    return original

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
def load_data(path, split=1):
    try:
        # compute the number of character classes (should be 1623)
        # to preallocate an array to fill
        alphabets = os.listdir(path)
        n_classes = 0
        for alphabet in alphabets:
            n_classes += len(os.listdir(path + '/' + alphabet))
        data = np.zeros((n_classes*n_ex_per_class, im_size, im_size))
        print '\t%d classes to load' % n_classes
        
        count = 0
        for alphabet in alphabets:
            alpha_path = path + '/' + alphabet
            for char in os.listdir(alpha_path):
                char_path = alpha_path + '/' + char
                char_idx = int(os.listdir(char_path)[0].split('.')[0].split('_')[0]) - 1
                char_offset = count * n_ex_per_class # offset for the class in the array
                for im in os.listdir(char_path):
                    # for each character, get the numbering and place in the appropriate spot
                    ex_idx = char_offset + int(im.split('.')[0].split('_')[1]) - 1
                    data[ex_idx,:,:] = load_image(char_path+'/'+im)
                count += 1
            if not (count % int(n_classes/10)):
                print '\tFinished %d classes' % count
            
    except Exception as e:
        pdb.set_trace()
    return data, n_classes

'''
Data will be arranged as follows

[ episode 1 ] -->   [ S ] total ((k + kb), im_size, im_size)
[ episode 2 ]       [ B ]   or ((k + kb), 1) for labels
    ...
[ episode N ]

'''
def create_episodes(data, n_episodes):
    try:
        k, kB = args.k, args.kB
        n_classes = data.shape[0] / n_ex_per_class
        n_examples = k + kB # n examples per class per episode
        base_bat_offset = args.N * k
        inputs = np.zeros((n_episodes, args.N * n_examples, im_size, im_size))
        outputs = np.zeros((n_episodes, args.N * n_examples, 1))

        # presample the N classes for each episode, including rotations classes (hence x4)
        # then for each class in the episode sample k+kB examples
        for i in xrange(n_episodes):
            episode_classes = np.random.choice(4*n_classes, args.N, replace=False)
            for j,c in enumerate(classes):
                n_rots = c / n_classes
                base_class_offset = c - (n_classes*n_rots)
                exs = np.random.choice(n_ex_per_class, n_examples, replace=False) + \
                        (base_class_offset * n_ex_per_class)
                set_offset = j*k
                bat_offset = base_bat_offset + j*kB
                for m in xrange(k): # TODO I'm sure there's a cleaner numpy way to do this
                    inputs[i,set_offset:set_offset+m,:,:] = np.rot90(data[exs[m],:,:], n_rots)
                    outputs[i,set_offset:set_offset+m,:] = c #np.ones((k,1))*c
                for m in xrange(kB):
                    inputs[i,bat_offset:bat_offset+m,:,:] = np.rot90(data[exs[k+m],:,:], n_rots)
                    outputs[i,bat_offset:bat_offset+m,:] = c #np.ones((kB,1))*c
    except Exception as e:
        pdb.set_trace()
    return inputs, outputs

def create_shards(data, n_episodes, n_shards, split):
    for i in xrange(n_shards):
        with h5py.File(args.out_path + split + "_%d.hdf5" % (i+1), 'w') as f:
            ins, outs = create_episodes(data, n_episodes)
            f['ins'] = ins
            f['outs'] = outs
        del ins, outs
        print '\t%d..' % (i+1),
    print '\n',


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_path', help='path to Omniglot data', type=str, default='/n/rush_lab/data/omniglot/small')
    parser.add_argument('--load_data_from', help='optional path to hdf5 file containing loaded data', type=str, default='')
    parser.add_argument('--save_data_to', help='optional path to save hdf5 file containing loaded data', type=str, default='')
    parser.add_argument('--out_path', help='path to folder to contain output files', type=str, default='')
    parser.add_argument('--n_tr_shards', help='number of shards to split tr data amongst', type=int, default=1)
    parser.add_argument('--n_val_shards', help='number of shards to split val data amongst', type=int, default=1)
    parser.add_argument('--n_te_shards', help='number of shards to split te data amongst', type=int, default=1)

    parser.add_argument('--N', help='number of unique classes per episode', type=int, default=5)
    parser.add_argument('--k', help='number of examples per class in the support set', type=int, default=1)
    parser.add_argument('--kB', help='number of examples per class in the batch', type=int, default=1) # note: this may not be what they do in paper
    parser.add_argument('--n_tr_episodes', help='number of tr episodes per shard', type=int, default=10000) # about 70s / 10k
    parser.add_argument('--n_val_episodes', help='number of val episodes per shard', type=int, default=10000)
    parser.add_argument('--n_te_episodes', help='number of te episodes per shard', type=int, default=10000)
    parser.add_argument('--splits', help='string containing fractions for train, validation, test sets respectively, e.g. .8,.1,.1', type=str, default='.75,.1,.1')
    parser.add_argument('--im_dim', help='dim of image along a side', type=int, default=28)
    args = parser.parse_args(arguments)
    
    # load all the examples for every class
    print 'Loading data...'
    global im_size
    if args.load_data_from:
        print '\tReading data from %s' % args.load_data_from
        f = h5py.File(args.load_data_from, 'r')
        data = f['data'][:]
        n_classes =  f['n_classes'][:][0]
        f.close()
        im_size = data.shape[1]
    else:
        print '\tReading data from images'
        im_size = args.im_dim
        bg_data, n_bg_classes  = load_data(args.data_path + '/' + 'images_background')
        eval_data, n_eval_classes = load_data(args.data_path + '/' + 'images_evaluation')

        n_classes = n_bg_classes + n_eval_classes
        data = np.zeros((bg_data.shape[0]+eval_data.shape[0], im_size, im_size))
        data[:bg_data.shape[0]] = bg_data
        data[bg_data.shape[0]:] = eval_data

        if args.save_data_to:
            with h5py.File(args.save_data_to, 'w') as f:
                f['data'] = data
                f['n_classes'] = np.array([n_classes], dtype=np.int32)
        print '\tSaved loaded data to %s' % args.save_data_to

    break_pts = [(n_ex_per_class) * int(n_classes * float(x)) for x in args.splits.split(',')]
    tr_data = data[:break_pts[0]]
    val_data = data[break_pts[0]:break_pts[0]+break_pts[1]]
    te_data = data[break_pts[0]+break_pts[1]:]
    print '\tData loaded!'
    pdb.set_trace()
    print '\tSplit sizes: %d, %d, %d' % (tr_data.shape[0]/(n_ex_per_class), val_data.shape[0]/(n_ex_per_class), te_data.shape[0]/(n_ex_per_class))

    # augment data
    #augment()

    # create episodes
    if args.out_path and args.out_path[-1] != '/':
        args.out_path += '/'
    print 'Creating data...'
    create_shards(tr_data, args.n_tr_episodes, args.n_tr_shards, "tr")
    print '\tFinished training data'
    create_shards(val_data, args.n_val_episodes, args.n_val_shards, "val")
    print '\tFinished validation data'
    create_shards(te_data, args.n_te_episodes, args.n_te_shards, "te")
    print '\tFinished test data'
    with h5py.File(args.out_path + "params.hdf5", 'w') as f:
        f['N'] = np.array([args.N], dtype=np.int32)
        f['k'] = np.array([args.k], dtype=np.int32)
        f['kB'] = np.array([args.kB], dtype=np.int32)
        f['n_tr_shards'] = np.array([args.n_tr_shards], dtype=np.int32)
        f['n_val_shards'] = np.array([args.n_val_shards], dtype=np.int32)
        f['n_te_shards'] = np.array([args.n_te_shards], dtype=np.int32)
    print 'Done!'

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
