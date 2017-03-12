import os
import sys
import argparse
import numpy as np
from scipy.misc import imread, imresize
import h5py
import pdb

n_ex_per_class = 20
counter = 0
total = 0

'''

Due to resizing, images are no longer purely black and white
Rounding to 0 or 1 based on a threshold seems ok
Threshold arbitrarily chosen as .5

'''
def load_image(f):
    global counter, total
    im = imread(f, flatten=True)
    if args.resize > 0:
        im = np.asarray(imresize(im, size=(args.resize, args.resize)), dtype=np.float32) / 255.
        args.im_dim = args.resize
    im = np.expand_dims(im, axis = 0)
    inverted = 1. - im
    max_val = np.max(inverted)
    if max_val > 0.:
        inverted /= max_val
        if max_val < 1.:
            counter += 1
    if args.thresh > 0:
        return (inverted / thresh).astype(int)
    else:
        total += 1
        return inverted

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
    global counter, total
    try:
        # compute the number of character classes (should be 1623)
        # to preallocate an array to fill
        alphabets = os.listdir(path)
        n_classes = 0
        for alphabet in alphabets:
            n_classes += len(os.listdir(path + '/' + alphabet))
        data = np.zeros((n_classes, n_ex_per_class, args.n_channels, args.im_dim, args.im_dim))
        print '\t%d classes to load' % n_classes
        
        count = 0
        for alphabet in alphabets:
            alpha_path = path + '/' + alphabet
            for char in os.listdir(alpha_path):
                char_path = alpha_path + '/' + char
                char_idx = int(os.listdir(char_path)[0].split('.')[0].split('_')[0]) - 1
                for im in os.listdir(char_path):
                    # for each character, get the numbering and place in the appropriate spot
                    ex_idx = int(im.split('.')[0].split('_')[1]) - 1
                    data[count,ex_idx,:,:] = load_image(char_path+'/'+im)
                count += 1
            if not (count % int(n_classes/10)):
                print '\tFinished %d classes\n' % count

    except Exception as e:
        pdb.set_trace()
    return data, n_classes

def augment(data):
    try:
        n_data = data.shape[0]
        augmented = np.zeros((n_data*4, n_ex_per_class, args.n_channels, args.im_dim, args.im_dim))
        for i, char in enumerate(data):
            for j,ex in enumerate(char):
                for k in xrange(4):
                    augmented[k*n_data+i, j,] = np.rot90(ex.transpose([2,1,0]), k).transpose([2,1,0])
    except Exception as e:
        pdb.set_trace()
    return augmented

'''

Data will be arranged as follows: n_episodes x (N*k + kb) x im_size x im_size

[ episode 1 ] -->   [ S ] total ((k + kb), im_size, im_size)
[ episode 2 ]       [ B ]   or ((k + kb), 1) for labels
    ...
[ episode N ]

'''
def create_oneshot_episodes(data, n_episodes):
    try:
        k, kB = args.k, args.kB
        n_classes = data.shape[0] 
        base_bat_offset = args.N * k
        inputs = np.zeros((n_episodes, (args.N*k)+kB, args.n_channels, args.im_dim, args.im_dim))
        outputs = np.zeros((n_episodes, (args.N*k)+kB, 1))

        # for each episode, sample N classes
        # then for each class sample k+kB examples
        for i in xrange(n_episodes):
            episode_classes = np.random.choice(n_classes, args.N, replace=False)
            batch_examples = np.random.multinomial(kB, [1./args.N]*args.N)
            batch_offset = 0
            for j,(c,n) in enumerate(zip(episode_classes, batch_examples)):
                exs = np.random.choice(n_ex_per_class, k+n, replace=False)
                set_offset = j*k
                bat_offset = base_bat_offset + batch_offset
                inputs[i,set_offset:set_offset+k,:,:] = data[c, exs[:k],:,:]
                inputs[i,bat_offset:bat_offset+n,:,:] = data[c, exs[k:],:,:]
                outputs[i,set_offset:set_offset+k,:] = np.ones((k,1)) * (j+1)
                outputs[i,bat_offset:bat_offset+n,:] = np.ones((n,1)) * (j+1)
                batch_offset += n
    except Exception as e:
        pdb.set_trace()
    return inputs, outputs

def create_baseline_episodes(data):
    try:
        inputs = np.zeros((data.shape[0]*n_ex_per_class, args.im_dim, args.im_dim))
        outputs = np.zeros((data.shape[0]*n_ex_per_class, 1))
        for j, char in enumerate(data):
            inputs[j*n_ex_per_class:(j+1)*n_ex_per_class, :, :] = char
            outputs[j*n_ex_per_class:(j+1)*n_ex_per_class, :] = np.ones((n_ex_per_class, 1)) * (j+1)
        p = np.random.permutation(inputs.shape[0]) 
    except Exception as e:
        pdb.set_trace()
    return inputs[p], outputs[p]

def create_shards(data, n_episodes, n_shards, split):
    if args.type == 'oneshot' or split != "tr":
        for i in xrange(n_shards):
            with h5py.File(args.out_path + split + "_%d.hdf5" % (i+1), 'w') as f:
                ins, outs = create_oneshot_episodes(data, n_episodes)
                f['ins'] = ins
                f['outs'] = outs
            del ins, outs
            print '\t%d..' % (i+1)
    else:
        assert(args.type == 'baseline' and split == "tr")
        with h5py.File(args.out_path + split + "_%d.hdf5" % 1, 'w') as f:
            ins, outs = create_baseline_episodes(data)
            f['ins'] = ins
            f['outs'] = outs
        del ins, outs
        print '\t%d..' % 1

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_path', help='path to Omniglot data', type=str, default='/n/rush_lab/data/omniglot/small')
    parser.add_argument('--load_data_from', help='optional path to hdf5 file containing loaded data', type=str, default='')
    parser.add_argument('--save_data_to', help='optional path to save hdf5 file containing loaded data', type=str, default='')
    parser.add_argument('--out_path', help='path to folder to contain output files', type=str, default='')

    parser.add_argument('--type', help='type of classifier data will train (oneshot or baseline)', type=str, default='oneshot')
    parser.add_argument('--n_tr_shards', help='number of shards to split tr data amongst', type=int, default=1)
    parser.add_argument('--n_val_shards', help='number of shards to split val data amongst', type=int, default=1)
    parser.add_argument('--n_te_shards', help='number of shards to split te data amongst', type=int, default=1)
    parser.add_argument('--augment', help='1 if augment with rotations', type=int, default=1)
    parser.add_argument('--reuse_test', help='1 if resue test classes for validation', type=int, default=1)

    parser.add_argument('--N', help='number of unique classes per episode', type=int, default=5)
    parser.add_argument('--k', help='number of examples per class in the support set', type=int, default=1)
    parser.add_argument('--kB', help='number of examples per class in the batch', type=int, default=10) 
    parser.add_argument('--n_tr_episodes', help='number of tr episodes per shard', type=int, default=5000) # about 70s / 10k
    parser.add_argument('--n_val_episodes', help='number of val episodes per shard', type=int, default=500)
    parser.add_argument('--n_te_episodes', help='number of te episodes per shard', type=int, default=500)
    parser.add_argument('--n_tr_classes', help='number of classes (before augmentation) for training, \
            remaining classes are split evenly among validation and test', type=int, default=1200)
    
    parser.add_argument('--im_dim', help='dim of image along a side', type=int, default=28)
    parser.add_argument('--n_channels', help='number of image channels', type=int, default=1)
    parser.add_argument('--thresh', help='threshold (in (0,1)) for image binarization, 0 for none', type=float, default=0.)
    parser.add_argument('--resize', help='dimension (along a side) to resize to, 0 for none', type=int, default=0)
    args = parser.parse_args(arguments)
    
    # load all the examples for every class
    print 'Loading data...'
    if args.load_data_from:
        print '\tReading data from %s' % args.load_data_from
        f = h5py.File(args.load_data_from, 'r')
        data = f['data'][:]
        n_classes =  f['n_classes'][:][0]
        f.close()
        args.im_dim = data.shape[-1]
    else:
        print '\tReading data from images'
        bg_data, n_bg_classes  = load_data(args.data_path + '/' + 'images_background')
        eval_data, n_eval_classes = load_data(args.data_path + '/' + 'images_evaluation')

        n_classes = n_bg_classes + n_eval_classes
        data = np.zeros((bg_data.shape[0]+eval_data.shape[0], n_ex_per_class, args.n_channels, args.im_dim, args.im_dim))
        data[:bg_data.shape[0]] = bg_data
        data[bg_data.shape[0]:] = eval_data

        if args.save_data_to:
            with h5py.File(args.save_data_to, 'w') as f:
                f['data'] = data
                f['n_classes'] = np.array([n_classes], dtype=np.int32)
        print '\tSaved loaded data to %s' % args.save_data_to

    #data -= .5
    #data -= np.mean(data, axis=(0,1))
    #data /= np.std(data, axis=(0,1))

    np.random.shuffle(data)
    if not args.reuse_test:
        split_pt = args.n_tr_classes + (n_classes - args.n_tr_classes)/2
        tr_data = data[:args.n_tr_classes]
        val_data = data[args.n_tr_classes:split_pt]
        te_data = data[split_pt:]
    else:
        tr_data = data[:args.n_tr_classes]
        val_data = te_data = data[args.n_tr_classes:]
    print '\tData loaded!'
    print '\tSplit sizes: %d, %d, %d' % (tr_data.shape[0], val_data.shape[0], te_data.shape[0])

    if args.augment:
        tr_data = augment(tr_data)

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
        f['n_classes'] = np.array([tr_data.shape[0]], dtype=np.int32)
    print 'Done!'

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
