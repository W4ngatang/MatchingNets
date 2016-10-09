require 'torch'
require 'nn'

package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua"
require 'data'
require 'match-nets'
require 'MapTable'

require 'nngraph'
require 'hdf5'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a matching network')
cmd:text('Options')

cmd:option('-seed', 42, 'random seed')
cmd:option('-gpuid', -1, '>=0 if use CUDA')
cmd:option('-cudnn', 0, '1 if use cudnn')

-- Input/Output options
cmd:option('-datafile', 'test.hdf5', 'path to hdf5 data file')

-- Model options --
cmd:option('-init_scale', .05, 'scale of initial parameters')
cmd:option('-embed_fn', '', 'Function to embed inputs')
cmd:option('-FCE', 0, '1 if use FCE, 0 otherwise')

-- CNN options --
cmd:option('-n_modules', 4, 'number of convolutional units to stack')
cmd:option('-n_kernels', 64, 'number of convolutional filters')
cmd:option('-conv_width', 3, 'convolution filter width')
cmd:option('-conv_height', 3, 'convolution filter height')
cmd:option('-pool_weight', 2, 'max pooling filter width')
cmd:option('-pool_height', 2, 'max pooling filter height')

-- Training options --
cmd:option('-n_epochs', 1, 'number of training epochs')
cmd:option('-learning_rate', 1, 'initial learning rate')
cmd:option('-batch_size', 1, 'number of episodes per batch')
cmd:option('-max_grad_norm', 5, 'maximum gradient value')

function train(model, data)
    local params, grad_params = model:getParameters()
    params:uniform(-opt.init_scale, opt.init_scale)
    local timer = torch.Timer()
    local last_loss = 1e9

    for epoch = 1, opt.n_epochs do
        print("Epoch", epoch)
        timer:reset()
        local total_loss = 0
        for i = 1, data.n_batches do -- TODO batching
            grad_params:zero()

            local episode = data[i]
            local inputs, targs = episode[1], episode[2]
            local outs = model:forward(inputs)
            local loss = crit:forward(outs, targs)
            total_loss = total_loss + loss
            local grad_loss = crit:backward(outs, targs)
            model:backward(inputs, deriv)
            local grad_norm = grad_params:norm()
            if grad_norm > opt.max_grad_norm then
                grad_params:mul(opt.max_grad_norm / grad_norm)
            end
            params:add(grad_params:mul(-opt.learning_rate))

            if i % 100 == 0 then
                print(i, data:size())
            end
        end
        print("Training time " .. timer:time().real .. ' seconds')

        -- TODO validation
    end

end

function evaluate(model, data)

end

function main()
    opt = cmd:parse(arg)
    torch.manualSeed(opt.seed)

    if opt.gpuid >= 0 then
        print('Using CUDA on GPU ' .. opt.gpuid .. '...')
        require 'cutorch'
        require 'cunn'
        if opt.cudnn == 1 then
            assert(opt.gpuid >= 0, 'GPU must be used if using cudnn')
            print('\tUsing cudnn...')
            require 'cudnn'
        end
        cutorch.setDevice(opt.gpuid+1)
        cutorch.manualSeed(opt.seed)
   end

   -- load data
   print('Loading data...')
   local f = hdf5.open(opt.datafile, 'r')
   local tr_ins = f:read('tr_ins'):all()
   local tr_outs = f:read('tr_outs'):all()
   local te_ins = f:read('te_ins'):all()
   local te_outs = f:read('te_outs'):all()
   opt.k = f:read('k'):all()[1]
   opt.N = f:read('N'):all()[1]
   opt.kB = f:read('kB'):all()[1]
   tr_data = data(opt, {tr_ins, tr_outs})
   te_data = data(opt, {te_ins, te_outs})
   print('\tData loaded!')
   
   -- build model
   print('Building model...')
   model = make_matching_net(opt)
   if opt.gpuid >= 0 then
       model = model:cuda()
    end
    print('\tModel built!')

   -- train
   print('Starting training...')
   train(model, tr_data)

   -- evaluate
   evaluate(model, te_data)
end

main()
