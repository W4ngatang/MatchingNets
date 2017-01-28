require 'torch'
require 'nn'
require 'optim'
require 'graph'

-- debugging library because lua doesn't seem to have one
package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'

require 'utils'
require 'data'
require 'data_baseline'
require 'match_nets'
require 'baseline'
require 'IndexAdd'

require 'Log2'

require 'nngraph'
require 'hdf5'
gModule = torch.getmetatable('nn.gModule')

function gModule:share(toShare, ...)
    for i, node in ipairs(self.forwardnodes) do
        if node.data.module then
            node.data.module:share(toShare.forwardnodes[i].data.module,...)
        end
    end
end

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a matching network')
cmd:text('Options')

cmd:option('--seed', 42, 'random seed')
cmd:option('--gpuid', 1, '>0 if use CUDA')
cmd:option('--cudnn', 1, '1 if use cudnn')

-- Input/Output options
cmd:option('--datafile', '', 'path to folder containing data files')
cmd:option('--data_folder', '', 'path to folder containing data files')
cmd:option('--logfile', '', 'file to log messages to')
cmd:option('--predfile', '', 'file to print test predictions to')
cmd:option('--print_freq', 5, 'how often to print training messages')

-- Model options --
cmd:option('--model', 'matching-net', 'model to use (matching-net or baseline')
cmd:option('--init_scale', .05, 'scale of initial parameters')
cmd:option('--init_dist', 'uniform', 'distribution to draw  initial parameters')
cmd:option('--share_embed', 0, '1 if share parameters between embedding functions')
cmd:option('--bn_eps', 1e-3, 'epsilson constant for batch normalization')
cmd:option('--bn_momentum', 0.1, 'momentum term in batch normalization')
cmd:option('--bn_affine', true, 'affine parameters in batch normalization')

-- CNN options --
cmd:option('--n_modules', 4, 'number of convolutional units to stack')
cmd:option('--n_kernels', 64, 'number of convolutional filters')
cmd:option('--nonlinearity', 'relu', 'nonlinearity to use')
cmd:option('--conv_width', 3, 'convolution filter width')
cmd:option('--conv_height', 3, 'convolution filter height')
cmd:option('--conv_pad', 1, 'convolutional padding')
cmd:option('--pool_ceil', 0, '1 if ceil in pooling dimension, else floor')
cmd:option('--pool_width', 2, 'max pooling filter width')
cmd:option('--pool_height', 2, 'max pooling filter height')
cmd:option('--pool_pad', 0, 'max pool padding')

-- Training options --
cmd:option('--n_epochs', 10, 'number of training epochs')
cmd:option('--optimizer', 'adagrad', 'optimizer to use (from optim)')
cmd:option('--learning_rate', .001, 'initial learning rate')
cmd:option('--learning_rate_decay', .00, 'learning rate decay')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--weight_decay', 0, 'weight decay')
cmd:option('--rho', .95, 'Adadelta interpolation parameter')
cmd:option('--beta1', .9, 'Adam beta1 parameter')
cmd:option('--beta2', .999, 'Adam beta2 parameter')
cmd:option('--batch_size', 1, 'number of episodes per batch')
cmd:option('--max_grad_norm', 0, 'maximum gradient value')

cmd:option('--debug', 0, '1 if stop for debug after 20 epochs')
cmd:option('--debug_after', 25, 'number of epochs after which to activate debugger')

function log(file, msg)
    print(msg)
    file:write(msg .. "\n")
end

function main()
    opt = cmd:parse(arg)
    torch.manualSeed(opt.seed)
    log_fh = io.open(opt.logfile, 'w')
    if opt.gpuid > 0 then
        log(log_fh, "Using CUDA on GPU " .. opt.gpuid .. "...")
        require 'cutorch'
        require 'cunn'
        if opt.cudnn == 1 then
            log(log_fh, "\tUsing cudnn...")
            require 'cudnn'
        end
        cutorch.setDevice(opt.gpuid)
        cutorch.manualSeed(opt.seed)
    end
    
    log(log_fh, 'Loading parameters...')
    local f = hdf5.open(opt.data_folder .. 'params.hdf5', 'r')
    opt.n_tr_shards = f:read('n_tr_shards'):all()[1]
    opt.n_val_shards = f:read('n_val_shards'):all()[1]
    opt.n_te_shards = f:read('n_te_shards'):all()[1]
    opt.k = f:read('k'):all()[1]
    opt.N = f:read('N'):all()[1]
    opt.kB = f:read('kB'):all()[1]
    if opt.model == 'baseline' then
        opt.n_classes = f:read('n_classes'):all()[1]
    end
    log(log_fh, '\tTraining with ' .. opt.n_tr_shards .. ' shards...')

    -- build model
    log(log_fh, 'Building model...')
    if opt.model == 'matching-net' then
        model = MatchingNetwork(opt, log_fh)
    elseif opt.model == 'baseline' then
        model = Baseline(opt, log_fh)
    end
    log(log_fh, '\tModel built!')
    collectgarbage()

    -- train
    log(log_fh, 'Starting training...')
    model:train()
    collectgarbage()

    -- evaluate
    log(log_fh, "Test accuracy: " .. model:evaluate("te"))

    -- cleanup
    log_fh:close()
end

main()
