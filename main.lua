require 'torch'
require 'nn'
require 'nngraph'
require 'hdf5'

require 'XXXX.models'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a matching network')
cmd:text('Options')
cmd:option('-seed', 42, 'random seed')

-- Model options --
cmd:option('-embed_fn', '', 'Function to embed inputs')
cmd:option('-FCE', 0, '1 if use FCE, 0 otherwise')

-- CNN options --
cmd:option('-n_kernels', 42, 'number of convolutional filters')
cmd:option('-conv_width', 3, 'convolution filter width')
cmd:option('-conv_height', 3, 'convolution filter height')
cmd:option('-pool_weight', 2, 'max pooling filter width')
cmd:option('-pool_height', 2, 'max pooling filter height')

function train()

end

function eval()

end

function main()
    opt = cmd:parse(arg)
    torch.manualSeed(opt.seed)

    if opt.gpuid >= 0 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        require 'cutorch'
        require 'cunn'
        if opt.cudnn == 1 then
            assert(opt.gpuid >= 0, 'GPU must be used if using cudnn')
            print('using cudnn...')
            require 'cudnn'
        end
        cutorch.setDevice(opt.gpuid+1)
        cutorch.manualSeed(opt.seed)
   end

   -- load data
   
   -- build model
end
