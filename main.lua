require 'torch'
require 'nn'

package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'
require 'data'
require 'match-nets'
require 'MapTable'
require 'IndexAdd'
require 'MM2'

require 'nngraph'
require 'hdf5'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a matching network')
cmd:text('Options')

cmd:option('--seed', 42, 'random seed')
cmd:option('--gpuid', 0, '>0 if use CUDA')
cmd:option('--cudnn', 0, '1 if use cudnn')

-- Input/Output options
cmd:option('--datafile', 'test.hdf5', 'path to hdf5 data file')
cmd:option('--logfile', 'log.log', 'file to log messages to')

-- Model options --
cmd:option('--init_scale', .05, 'scale of initial parameters')
cmd:option('--embed_fn', '', 'Function to embed inputs')
cmd:option('--match_fn', 'softmax', 'Function to score matches')
cmd:option('--FCE', 0, '1 if use FCE, 0 otherwise')

-- CNN options --
cmd:option('--n_modules', 4, 'number of convolutional units to stack')
cmd:option('--n_kernels', 64, 'number of convolutional filters')
cmd:option('--conv_width', 3, 'convolution filter width')
cmd:option('--conv_height', 3, 'convolution filter height')
cmd:option('--pool_width', 2, 'max pooling filter width')
cmd:option('--pool_height', 2, 'max pooling filter height')

-- Training options --
cmd:option('--n_epochs', 1, 'number of training epochs')
cmd:option('--learning_rate', 1, 'initial learning rate')
cmd:option('--batch_size', 1, 'number of episodes per batch')
cmd:option('--max_grad_norm', 5, 'maximum gradient value')

function log(file, msg)
    print(msg)
    file:write(msg .. "\n")
end

function train(model, crit, tr_data, val_data)
    local params, grad_params = model:getParameters()
    params:uniform(-opt.init_scale, opt.init_scale)
    local timer = torch.Timer()
    local last_score = evaluate(model, val_data)
    log(file, "Initial validation accuracy: " .. last_score)

    for epoch = 1, opt.n_epochs do
        log(file, "Epoch " ..epoch .. ", learning rate " .. opt.learning_rate )
        timer:reset()
        local total_loss = 0
        for i = 1, tr_data.n_batches do -- TODO batching
            grad_params:zero()
            local episode = tr_data[i]
            local inputs, targs = episode[1], episode[2]
            local outs = model:forward(inputs)
            local loss = crit:forward(outs, targs)
            total_loss = total_loss + loss
            local grad_loss = crit:backward(outs, targs)
            model:backward(inputs, grad_loss)
            local grad_norm = grad_params:norm()
            if grad_norm > opt.max_grad_norm then
                grad_params:mul(opt.max_grad_norm / grad_norm)
            end
            params:add(grad_params:mul(-opt.learning_rate))
            if i % (tr_data.n_batches/2) == 0 then
                log(file, "\t  Completed " .. i/tr_data.n_batches*100 .. "% in " ..timer:time().real .. " seconds")
            end
        end
        log(file, "\tTraining time: " .. timer:time().real .. " seconds")
        timer:reset()
        val_score = evaluate(model, val_data)
        log(file, "\tValidation time " .. timer:time().real .. " seconds")
        log(file, "\tLoss: " .. total_loss)
        log(file, "\tValidation accuracy: " .. val_score)
        if val_score < last_score then
            opt.learning_rate = opt.learning_rate/2
        end
        last_score = val_score
    end
end

function evaluate(model, data)
    local last_loss = 1e9
    local n_preds = 0
    local n_correct = 0
    for i = 1, data.n_batches do -- TODO batching
        local episode = data[i]
        local inputs, targs = episode[1], episode[2]
        local outs = model:forward(inputs)
        local maxes, preds = torch.max(outs, 2)
        if opt.gpuid > 0 then
            preds = preds:cuda()
        end
        n_correct = n_correct + torch.sum(torch.eq(preds, targs))
        n_preds = n_preds + targs:nElement()
    end
    local accuracy = n_correct/n_preds
    return accuracy
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

   -- load data
   log(file, 'Loading data...')
   local f = hdf5.open(opt.datafile, 'r')
   local tr_ins = f:read('tr_ins'):all()
   local tr_outs = f:read('tr_outs'):all()
   local val_ins = f:read('val_ins'):all()
   local val_outs = f:read('val_outs'):all()
   opt.k = f:read('k'):all()[1]
   opt.N = f:read('N'):all()[1]
   opt.kB = f:read('kB'):all()[1]
   tr_data = data(opt, {tr_ins, tr_outs})
   val_data = data(opt, {val_ins, val_outs})
   log(file, '\tData loaded!')
   
   -- build model
   log(file, 'Building model...')
   model, crit = make_matching_net(opt)
   if opt.gpuid > 0 then
       model = model:cuda()
       crit = crit:cuda()
    end
   log(file, '\tModel built!')

   -- train
   log(file, 'Starting training...')
   train(model, crit, tr_data, val_data)
   tr_data = nil
   val_data = nil
   collectgarbage()

   -- evaluate
   local te_ins = f:read('te_ins'):all()
   local te_outs = f:read('te_outs'):all()
   te_data = data(opt, {te_ins, te_outs})
   test_acc = evaluate(model, te_data)
   log(file, "Test accuracy: " .. test_acc)
   io.close(file)
end

main()
