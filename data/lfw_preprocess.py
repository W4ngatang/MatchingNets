import os
import sys
import argparse
import h5py
import pdb
import pickle
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize, imsave

def load_data(path):
    '''
    Load data from raw images
    Inputs:
        - path to LFW data (structured with a directory per person
    Outputs:
        - person2id: dict from string name to id
        - id2ims: dict from id to images
        - n_classes: number of persons
    '''

    person2id = {}
    id2ims = {}
    for person in os.listdir(path):
        raw_ims = os.listdir(path+person)
        if len(raw_ims) >= args.min_ims:
            ims = []
            person_id = len(person2id)
            person2id[person] = person_id
            for raw_im in raw_ims:
                ims.append((np.array(Image.open(path+person+'/'+raw_im)) / 255.).transpose([2,1,0]))
            id2ims[person_id] = np.array(ims)

    return person2id, id2ims, len(id2ims)

def augment(data):
    '''
    Augment face data with
        - reflections
        - crops
    '''
    try:
        for person, ims in data.iteritems(): # probably better to do this while creating episodes
            data[person] = np.vstack((data[person], np.array([np.fliplr(im) for im in ims])))
    except Exception as e:
        pdb.set_trace()
    return augmented

def create_oneshot_episodes(data, n_episodes, reflections=0, crops=None):
    '''

    '''
    try:
        k, kB, N = args.k, args.kB, args.N
        n_classes = len(data)
        base_bat_offset = N * k
        inputs = np.zeros((n_episodes, (N*k)+kB, args.n_channels, args.im_dim, args.im_dim))
        outputs = np.zeros((n_episodes, (N*k)+kB, 1))

        # for each episode, sample N classes
        # then for each class sample k+kB examples
        for i in xrange(n_episodes):
            episode_classes = np.random.choice(data.keys(), args.N, replace=False)
            #batch_examples = np.random.multinomial(kB, [1./args.N]*args.N)
            batch_examples = np.array([kB / N] * N) # not enough images per class, so fix 2 ims per class in batch
            batch_offset = 0
            for j, (c,n) in enumerate(zip(episode_classes, batch_examples)):
                exs = np.random.choice(data[c].shape[0], k+n, replace=False)
                set_offset = j*k
                bat_offset = base_bat_offset + batch_offset
                if reflections:
                    reflect_berns = np.random.randint(2, size=k+n)
                    inputs[i,set_offset:set_offset+k] = np.array([np.fliplr(data[c][p].transpose([2,1,0])).transpose([2,1,0]) if q else data[c][p] \
                            for (p,q) in zip(exs[:k], reflect_berns[:k])])
                    inputs[i,bat_offset:bat_offset+n] = np.array([np.fliplr(data[c][p].transpose([2,1,0])).transpose([2,1,0]) if q else data[c][p] \
                            for (p,q) in zip(exs[k:], reflect_berns[k:])])
                else:
                    inputs[i,set_offset:set_offset+k] = data[c][exs[:k]]
                    inputs[i,bat_offset:bat_offset+n] = data[c][exs[k:]]
                outputs[i,set_offset:set_offset+k] = np.ones((k,1)) * (j+1)
                outputs[i,bat_offset:bat_offset+n] = np.ones((n,1)) * (j+1)
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
                if split == "tr":
                    ins, outs = create_oneshot_episodes(data, n_episodes, args.reflections, args.crops)
                else:
                    ins, outs = create_oneshot_episodes(data, n_episodes)
                f['ins'] = ins
                f['outs'] = outs
            del ins, outs
            print '\t%d..' % (i+1)
            sys.stdout.flush()
    else: # TODO I haven't tested the baseline stuff
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
    parser.add_argument('--normalize', help='1 if normalize', type=int, default=1)
    parser.add_argument('--reflections', help='1 if augment with reflections', type=int, default=1)
    parser.add_argument('--crops', help='1 if augment with crops', type=int, default=1)
    parser.add_argument('--reuse_test', help='1 if resue test classes for validation', type=int, default=1)
    parser.add_argument('--min_ims', help='minimum number of images for a person to be used', type=int, default=3)

    parser.add_argument('--N', help='number of unique classes per episode', type=int, default=5)
    parser.add_argument('--k', help='number of examples per class in the support set', type=int, default=1)
    parser.add_argument('--kB', help='number of examples per class in the batch', type=int, default=10) 
    parser.add_argument('--n_tr_episodes', help='number of tr episodes per shard', type=int, default=5000) # about 70s / 10k
    parser.add_argument('--n_val_episodes', help='number of val episodes per shard', type=int, default=500)
    parser.add_argument('--n_te_episodes', help='number of te episodes per shard', type=int, default=500)
    parser.add_argument('--n_tr_classes', help='fraction of classes (before augmentation) for training, \
            remaining classes are split evenly among validation and test', type=int, default=.8)
    
    parser.add_argument('--im_dim', help='dim of image along a side', type=int, default=250)
    parser.add_argument('--n_channels', help='number of image channels', type=int, default=3)
    parser.add_argument('--thresh', help='threshold (in (0,1)) for image binarization, 0 for none', type=float, default=0.)
    parser.add_argument('--resize', help='dimension (along a side) to resize to, 0 for none', type=int, default=0)
    args = parser.parse_args(arguments)
    
    print 'Loading data...'
    if args.load_data_from:
        print '\tReading data from %s' % args.load_data_from
        f = h5py.File(args.load_data_from, 'r')
        all_ims = f['all_ims'][:]
        meta =  f['meta'][:]
        f.close()
        id2ims = {}
        for i, (id, start, size) in enumerate(meta):
            id2ims[id] = all_ims[start:start+size]
        if not args.normalize:
            del all_ims, meta
        n_classes = len(id2ims)
    else:
        print '\tReading data from images'
        if args.data_path[-1] != '/':
            args.data_path += '/'
        person2id, id2ims, n_classes = load_data(args.data_path)
        n_ims = sum([v.shape[0] for v in id2ims.values()])
        print '\tLoaded %d images for %d classes' % (n_ims, n_classes)

        if args.save_data_to:
            all_ims = np.zeros((n_ims, args.n_channels, args.im_dim, args.im_dim))
            meta = np.zeros((n_classes, 3))
            start = 0
            for i, (person, ims) in enumerate(id2ims.iteritems()):
                all_ims[start:start+ims.shape[0]] = ims
                meta[i] = [person, start, ims.shape[0]]
                start += ims.shape[0]
            assert start == n_ims

            with h5py.File(args.save_data_to, 'w') as f:
                f['all_ims'] = all_ims
                f['meta'] = meta
            
            if not args.normalize:
                del all_ims, meta
            print '\tSaved loaded data to %s' % args.save_data_to

    if args.normalize:
        if 'all_ims' not in locals():
            all_ims = np.zeros((n_ims, args.n_channels, args.im_dim, args.im_dim))
            start = 0
            for person, ims in id2ims.iteritems():
                all_ims[start:start+ims.shape[0]] = ims
                start += ims.shape[0]
            assert start == n_ims
        mean = np.mean(all_ims, axis = 0)
        std = np.std(all_ims, axis = 0)
        for person, ims in id2ims.iteritems():
            id2ims[person] = (ims - mean) / std
        del all_ims

    class_splits = id2ims.keys()
    np.random.shuffle(class_splits)
    tr_split_pt = int(args.n_tr_classes * n_classes)
    if not args.reuse_test: # in original Matching Nets, they use same classes for val and test
        val_split_pt = int(n_classes * (args.n_tr_classes + (1. - args.n_tr_classes)/2))
        tr_data = {i: id2ims[i] for i in class_splits[:tr_split_pt]}
        val_data = {i: id2ims[i] for i in class_splits[tr_split_pt:val_split_pt]}
        te_data = {i: id2ims[i] for i in class_splits[val_split_pt:]}
    else:
        tr_data = {i: id2ims[i] for i in class_splits[:tr_split_pt]}
        val_data = te_data = {i: id2ims[i] for i in class_splits[tr_split_pt:]}
    #del id2ims
    print '\tData loaded!'
    print '\tSplit sizes: %d, %d, %d' % (len(tr_data), len(val_data), len(te_data))

    #if args.augment:
    #    tr_data = augment(tr_data)

    # create episodes
    if args.out_path[-1] != '/':
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
        f['n_classes'] = np.array([n_classes], dtype=np.int32)
        if args.normalize:
            f['mean'] = mean
            f['std'] = std
    print 'Done!'

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
