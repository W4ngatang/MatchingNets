require 'torch'
require 'nn'
require 'optim'
require 'graph'

package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'
require 'data'
require 'match-nets'
require 'MapTable'
require 'IndexAdd'
require 'MM2'
require 'Log2'

require 'nngraph'
require 'hdf5'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a matching network')
cmd:text('Options')

cmd:option('--seed', 42, 'random seed')
cmd:option('--gpuid', 1, '>0 if use CUDA')
cmd:option('--cudnn', 1, '1 if use cudnn')

-- Input/Output options
cmd:option('--datafile', '', 'path to folder containing data files')
cmd:option('--data_folder', '', 'path to folder containing data files')
cmd:option('--logfile', '', 'file to log messages to')
cmd:option('--print_freq', 5, 'how often to print training messages')

-- Model options --
cmd:option('--init_scale', .05, 'scale of initial parameters')
cmd:option('--init_dist', 'normal', 'distribution to draw  initial parameters')
cmd:option('--share_embed', 0, '1 if share parameters between embedding functions')
cmd:option('--match_fn', 'softmax', 'Function to score matches')
cmd:option('--embed_fn', '', 'Function to embed inputs')
cmd:option('--FCE', 0, '1 if use FCE, 0 otherwise')

-- CNN options --
cmd:option('--n_modules', 4, 'number of convolutional units to stack')
cmd:option('--n_kernels', 64, 'number of convolutional filters')
cmd:option('--conv_width', 3, 'convolution filter width')
cmd:option('--conv_height', 3, 'convolution filter height')
cmd:option('--pool_width', 2, 'max pooling filter width')
cmd:option('--pool_height', 2, 'max pooling filter height')

-- Training options --
cmd:option('--n_epochs', 10, 'number of training epochs')
cmd:option('--optimizer', 'adagrad', 'optimizer to use (from optim)')
cmd:option('--learning_rate', .001, 'initial learning rate')
cmd:option('--learning_rate_decay', .01, 'learning rate decay') -- maybe want .99
cmd:option('--momentum', .5, 'momentum')
cmd:option('--rho', .95, 'Adadelta interpolation parameter')
cmd:option('--halve_learning_rate', 0, '1 if halve learning rate if val score decreases between successive epochs')
cmd:option('--batch_size', 1, 'number of episodes per batch')
cmd:option('--max_grad_norm', 0, 'maximum gradient value')

function log(file, msg)
    print(msg)
    file:write(msg .. "\n")
end

function train(model, crit)
    local params, grad_params = model:getParameters()
    if opt.init_dist == 'normal' then
        params:normal(0, opt.init_scale)
    elseif opt.init_dist == 'uniform' then
        params:uniform(-opt.init_scale, opt.init_scale)
    else
        log(file, 'Unsupported distribution for initialization!')
        return
    end

    --[[ Optimization ]]--
    local inputs, targs, total_loss
    function feval(p)
        grad_params:zero()
        local outs = model:forward(inputs)
        local loss = crit:forward(outs, targs)
        total_loss = total_loss + loss
        local grad_loss = crit:backward(outs, targs)
        model:backward(inputs, grad_loss)
        local grad_norm = grad_params:norm()
        if opt.max_grad_norm > 0 and grad_norm > opt.max_grad_norm then
            grad_params:mul(opt.max_grad_norm / grad_norm)
        end
        return loss, grad_params
    end
    local optimize, optim_state
    log(file, "\tOptimizing with " .. opt.optimizer)
    if opt.optimizer == 'sgd' then
        optim_state = { -- NB: Siamese paper used layerwise LR and momentum
            learningRate = opt.learning_rate,
            weightDecay = 0,
            momentum = opt.momentum,
            learningRateDecay = opt.learning_rate_decay
        }
        optimize = optim.sgd
    elseif opt.optimizer == 'adagrad' then
        optim_state = {
            learningRate = opt.learning_rate
        }
        optimize = optim.adagrad
        log(file, "\t\twith learning rate " .. opt.learning_rate)
    elseif opt.optimizer == 'adadelta' then
        optim_state = {
            rho = opt.rho,
            eps = 1e-8
        }
        optimize = optim.adadelta
        log(file, "\t\twith rho " .. opt.rho)
    else
        error('Unknown optimizer!')
    end
    if opt.max_grad_norm > 0 then
        log(file, "\t\twith max gradient norm: " .. opt.max_grad_norm)
    end

    --[[ Training Loop ]]--
    local timer = torch.Timer()
    local last_score = evaluate(model, "val")
    local best_score = last_score
    log(file, "\tInitial validation accuracy: " .. last_score)
    for epoch = 1, opt.n_epochs do
        log(file, "Epoch " ..epoch)
        timer:reset()
        total_loss = 0
        for shard_n = 1, opt.n_tr_shards do
            local f = hdf5.open(opt.data_folder .. 'tr_' .. shard_n .. '.hdf5', 'r')
            local tr_ins = f:read('ins'):all()
            local tr_outs = f:read('outs'):all()
            local tr_data = data(opt, {tr_ins, tr_outs})
            for i = 1, tr_data.n_batches do
                local episode = tr_data[i]
                inputs, targs = episode[1], episode[2]
                optimize(feval, params, optim_state)
            end
            if shard_n % (10/opt.print_freq) == 0 then
                log(file, "\t  Completed " .. shard_n/opt.n_tr_shards*100 .. "% in " ..timer:time().real .. " seconds")
            end
            tr_ins = nil
            tr_outs = nil
            tr_data = nil
            collectgarbage()
        end
        log(file, "\tTraining time: " .. timer:time().real .. " seconds")
        timer:reset()
        val_score = evaluate(model, "val")
        log(file, "\tValidation time " .. timer:time().real .. " seconds")
        if opt.halve_learning_rate > 0 and val_score < last_score then
            optim_state['learningRate'] = optim_state['learningRate']/2
        end
        if val_score > best_score then
            -- TODO: save the model
            best_score = val_score
        end
        last_score = val_score
        log(file, "\tLoss: " .. total_loss .. ", Validation accuracy: " .. val_score .. ", Best accuracy: " .. best_score)
        --log(file, "\tValidation accuracy: " .. val_score)
        --log(file, "\tBest accuracy: " .. best_score)
    end
end

function evaluate(model, split)
    local n_preds = 0
    local n_correct = 0
    local n_shards = opt.n_te_shards
    local str_split = 'te_'
    if split == "val" then
        n_shards = opt.n_val_shards
        str_split = 'val_'
    end
    for shard_n = 1, n_shards do 
        local f = hdf5.open(opt.data_folder .. str_split .. shard_n .. '.hdf5', 'r')
        local sp_ins = f:read('ins'):all()
        local sp_outs = f:read('outs'):all()
        local sp_data = data(opt, {sp_ins, sp_outs})

        for i = 1, sp_data.n_batches do
            local episode = sp_data[i]
            local inputs, targs = episode[1], episode[2]
            local outputs = model:forward(inputs)
            local maxes, preds = torch.max(outputs, 2)
            if opt.gpuid > 0 then
                preds = preds:cuda()
            end
            n_correct = n_correct + torch.sum(torch.eq(preds, targs))
            n_preds = n_preds + targs:nElement()
        end
        ins = nil
        outs = nil
        sp_data = nil
        collectgarbage()
    end
    return n_correct/n_preds
end

function main()
    opt = cmd:parse(arg)
    torch.manualSeed(opt.seed)
    file = io.open(opt.logfile, 'w')
    if opt.gpuid > 0 then
        log(file, "Using CUDA on GPU " .. opt.gpuid .. "...")
        require 'cutorch'
        require 'cunn'
        if opt.cudnn == 1 then
            assert(opt.gpuid > 0, 'GPU must be used if using cudnn')
            log(file, "\tUsing cudnn...")
            require 'cudnn'
        end
        cutorch.setDevice(opt.gpuid)
        cutorch.manualSeed(opt.seed)
    end
    -- TODO append '/' to data_folder
    
    log(file, 'Loading parameters...')
    local f = hdf5.open(opt.data_folder .. 'params.hdf5', 'r')
    opt.n_tr_shards = f:read('n_tr_shards'):all()[1]
    opt.n_val_shards = f:read('n_val_shards'):all()[1]
    opt.n_te_shards = f:read('n_te_shards'):all()[1]
    opt.k = f:read('k'):all()[1]
    opt.N = f:read('N'):all()[1]
    opt.kB = f:read('kB'):all()[1]
    log(file, '\tTraining with ' .. opt.n_tr_shards .. ' shards...')

   -- build model
    log(file, 'Building model...')
    if opt.share_embed == 1 then
        log(file, '\tTying embedding function parameters...')
    end
    model, crit = make_matching_net(opt)
    --graph.dot(model.fg, 'match-net', 'my-match-net')
    if opt.gpuid > 0 then
        model = model:cuda()
        crit = crit:cuda()
    end
    log(file, '\tModel built!')
    collectgarbage()

    -- train
    log(file, 'Starting training...')
    train(model, crit)--, tr_data, val_data)
    collectgarbage()

   -- evaluate
    test_acc = evaluate(model, "te")
    log(file, "Test accuracy: " .. test_acc)
    io.close(file)
end

main()
