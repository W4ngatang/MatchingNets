require 'torch'
require 'nn'
require 'rnn'
require 'optim'
require 'graph'
require 'pl'

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
require 'Normalize2'
require 'MM2'

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

-- GPU and misc. --
cmd:option('--seed', 42, 'random seed')
cmd:option('--gpuid', 1, '>0 if use CUDA')
cmd:option('--cudnn', 1, '1 if use cudnn')

-- Input,Output options --
cmd:option('--datafile', '', 'path to folder containing data files')
cmd:option('--data_folder', '', 'path to folder containing data files')
cmd:option('--logfile', '', 'file to log messages to')
cmd:option('--predfile', '', 'file to print test predictions to')
cmd:option('--load_model_from', '', 'file to load best model from')
cmd:option('--save_model_to', '', 'file to save best model to')
cmd:option('--print_freq', 5, 'how often to print training messages')

-- Model options --
cmd:option('--model', 'matching-net', 'model to use (matching-net or baseline')
cmd:option('--init_scale', .05, 'scale of initial parameters')
cmd:option('--init_dist', 'uniform', 'distribution to draw initial parameters')
cmd:option('--load_params_from', '', 'file to load weights from')
cmd:option('--save_params_to', '', 'file to save weights to')
cmd:option('--embedding_fn', 'cnn', 'type of embedding function to use')
cmd:option('--share_embed', 1, '1 if share parameters between embedding functions')
cmd:option('--prototypes', 0, '1 if use class prototypes')
cmd:option('--contextual_f','', 'type of parameters to add after embedding functions')
cmd:option('--contextual_g','', 'type of parameters to add after embedding functions')
cmd:option('--init_fce','zero', 'how to initialize FCE hidden state and cells')
cmd:option('--L',5, 'number of rollout steps for FCE')
cmd:option('--bn_eps', 1e-3, 'epsilson constant for batch normalization')
cmd:option('--bn_momentum', 0.1, 'momentum term in batch normalization')
cmd:option('--bn_affine', true, 'affine parameters in batch normalization')

cmd:option('--d_emb', 100, 'word vector dimension or number of convolutional filters')
cmd:option('--pretrain_file', '', 'path to pretrained embeddings')


-- CNN options --
cmd:option('--n_channels', 1, 'number of initial image channels')
cmd:option('--im_dim', 64, 'image dimensions (assuming square)')
cmd:option('--n_modules', 4, 'number of convolutional units to stack')
cmd:option('--kernel_sizes', '3,3,3,3', 'sizes of convolutional filters')
cmd:option('--nonlinearity', 'relu', 'nonlinearity to use')
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
cmd:option('--baseline_batch_size', 25, 'number of episodes per batch for baseline model')
cmd:option('--max_grad_norm', 0, 'maximum gradient value')

cmd:option('--debug', 0, '1 if stop for debug after n epochs')
cmd:option('--debug_on', '', 'comma separated string of epochs to debug after')
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
    if opt.embedding_fn == 'bow' or opt.embedding_fn == 'lstm' then
        opt.vocab_size = f:read('vocab_size'):all()[1]
        opt.seq_len = f:read('seq_len'):all()[1]
    elseif opt.embedding_fn == 'none' then
        opt.d_emb = f:read('n_feats'):all()[1]
    end
    if opt.model == 'baseline' then
        opt.n_classes = f:read('n_classes'):all()[1]
    end
    log(log_fh, '\tTraining with ' .. opt.n_tr_shards .. ' shards...')

    log(log_fh, 'Building model...')
    if opt.load_model_from ~= '' then
        model = torch.load(opt.load_model_from)
    else
        if opt.model == 'matching-net' then
            model = MatchingNetwork(opt, log_fh)
        elseif opt.model == 'baseline' then
            model = Baseline(opt, log_fh)
        end
    end
    log(log_fh, '\tModel built!')
    collectgarbage()

    -- train
    log(log_fh, 'Starting training...')
    model:train(log_fh)
    collectgarbage()

    -- evaluate
    if opt.save_model_to ~= '' then
        model = nil
        collectgarbage()
        model = torch.load(opt.save_model_to)
    end
    log(log_fh, "Best validation accuracy: " .. model:evaluate("val"))
    log(log_fh, "Test accuracy: " .. model:evaluate("te"))

    -- cleanup
    log_fh:close()
end

main()
