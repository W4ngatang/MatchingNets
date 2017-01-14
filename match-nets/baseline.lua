package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'

local Baseline = torch.class("Baseline")

--
-- Various models used throughout experiments
--

function Baseline:__init(opt, log_fh)
    -- input is support set and labels, test datum
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- hat(x): B x im x im

    -- in: B x im x im
    --   unsqueeze -> B x 1 x im x im
    --   f -> B x 64 x 1 x 1
    --   squeeze -> B x 64
    --   linear -> B x n_classes
    --   LSM -> B x n_classes
    -- out: B x n_classes
    local f = make_cnn(opt)
    local embed_f = nn.Squeeze()(f(nn.Unsqueeze(2)(inputs[1])))
    local linear = nn.Linear(opt.n_kernels, opt.n_classes)(embed_f)
    local outputs = {linear}
    local crit = nn.CrossEntropyCriterion()
    local model = nn.gModule(inputs, outputs)
    if opt.gpuid > 0 then
        model = model:cuda()
        crit = crit:cuda()
    end
    self.model = model
    self.crit = crit
    self.opt = opt
    self.log_fh = log_fh
end

function Baseline:train()
    local model = self.model
    local crit = self.crit
    local opt = self.opt
    local log_fh = self.log_fh

    --[[ Parameter Init ]]--
    params, grad_params = model:getParameters()
    if opt.init_dist == 'normal' then
        params:normal(0, opt.init_scale)
    elseif opt.init_dist == 'uniform' then
        params:uniform(-opt.init_scale, opt.init_scale)
    else
        log(log_fh, 'Unsupported distribution for initialization!')
        return
    end

    --[[ Optimization Setup ]]--
    log(log_fh, "\tOptimizing with " .. opt.optimizer)
    if opt.optimizer == 'sgd' then
        optim_state = { -- NB: Siamese paper used layerwise LR and momentum
            learningRate = opt.learning_rate,
            weightDecay = 0,
            momentum = opt.momentum,
            learningRateDecay = opt.learning_rate_decay
        }
        optimize = optim.sgd
        log(log_fh, "\t\twith learning rate " .. opt.learning_rate)
        log(log_fh, "\t\twith learning rate decay " .. opt.learning_rate_decay)
        log(log_fh, "\t\twith momentum " .. opt.momentum)
    elseif opt.optimizer == 'adagrad' then
        optim_state = {
            learningRate = opt.learning_rate
        }
        optimize = optim.adagrad
        log(log_fh, "\t\twith learning rate " .. opt.learning_rate)
        log(log_fh, "\t\twith learning rate decay " .. opt.learning_rate_decay)
    elseif opt.optimizer == 'adadelta' then
        optim_state = {
            rho = opt.rho,
            eps = 1e-8
        }
        optimize = optim.adadelta
        log(log_fh, "\t\twith rho " .. opt.rho)
    elseif opt.optimizer == 'adam' then
        optim_state = {
            learningRate = opt.learning_rate,
            learningRateDecay = opt.learning_rate_decay,
            beta1 = opt.beta1,
            beta2 = opt.beta2
        }
        optimize = optim.adam
        log(log_fh, "\t\twith learning rate " .. opt.learning_rate)
        log(file, "\t\twith learning rate decay " .. opt.learning_rate_decay)
        log(file, "\t\twith beta1 " .. opt.beta1)
        log(file, "\t\twith beta2 " .. opt.beta2)
    else
        error('Unknown optimizer!')
    end
    if opt.max_grad_norm > 0 then
        log(log_fh, "\t\twith max gradient norm: " .. opt.max_grad_norm)
    end

    --[[ Optimization Step ]]--
    local inputs, targs, total_loss, n_correct, n_preds, n_batches, tr_data
    function feval(p)
        grad_params:zero()
        local outs = model:forward(inputs)
        local loss = crit:forward(outs, targs)
        local maxes, preds = torch.max(outs, 2)
        if opt.gpuid > 0 then
            preds = preds:cuda()
        end
        n_correct = n_correct + torch.sum(torch.eq(preds, targs))
        n_preds = n_preds + targs:nElement()
        total_loss = total_loss + loss
        local grad_loss = crit:backward(outs, targs)
        model:backward(inputs, grad_loss)
        local grad_norm = grad_params:norm()
        if opt.max_grad_norm > 0 and grad_norm > opt.max_grad_norm then
            grad_params:mul(opt.max_grad_norm / grad_norm)
        end
        return loss, grad_params
    end

    --[[ Training Loop ]]--
    local timer = torch.Timer()
    local last_score = self:evaluate("val")
    local best_score = last_score
    local best_params = params:clone()
    log(log_fh, "\tInitial validation accuracy: " .. last_score)
    for epoch = 1, opt.n_epochs do
        model:training()
        log(log_fh, "Epoch " ..epoch)
        timer:reset()
        total_loss = 0
        n_correct = 0
        n_preds = 0
        n_batches = 0
        for shard_n = 1, opt.n_tr_shards do
            local f = hdf5.open(opt.data_folder .. 'tr_' .. shard_n .. '.hdf5', 'r')
            local tr_ins = f:read('ins'):all()
            local tr_outs = f:read('outs'):all()
            local tr_data = DataBaseline(opt, {tr_ins, tr_outs})
            for i = 1, tr_data.n_batches do
                local episode = tr_data[i]
                inputs, targs = episode[1], episode[2]
                optimize(feval, params, optim_state)
            end
            if shard_n % (10/opt.print_freq) == 0 then
                log(file, "\t  Completed " .. shard_n/opt.n_tr_shards*100 .. "% in " ..timer:time().real .. " seconds")
            end
            n_batches = n_batches + tr_data.n_batches
            tr_ins = nil
            tr_outs = nil
            tr_data = nil
            collectgarbage()
        end
        log(log_fh, "\tTraining time: " .. timer:time().real .. " seconds")
        timer:reset()
        
        if epoch > opt.debug_after then -- point for debugging
            flag = 1
        end

        val_score = self:evaluate("val")
        log(log_fh, "\tValidation time " .. timer:time().real .. " seconds")
        if opt.learning_rate_decay > 0 and opt.optimizer == 'adagrad' and val_score < last_score then
            optim_state['learningRate'] = optim_state['learningRate'] * opt.learning_rate_decay
            log(log_fh, "\tLearning rate decayed to " .. optim_state['learningRate'])
        end
        if val_score > best_score then
            best_score = val_score
            best_params:copy(params)
        end
        last_score = val_score
        log(log_fh, "\tLoss: " .. total_loss/n_batches .. ", training accuracy: " .. n_correct/n_preds .. ", Validation accuracy: " .. val_score .. ", Best accuracy: " .. best_score)
    end
    params:copy(best_params)
    print("Best validation accuracy: " .. self:evaluate("val"))

end

function Baseline:evaluate(split, fh)
    local model = self.model
    local crit = self.crit
    local opt = self.opt
    local log_fh = self.log_fh

    model:evaluate()
    local n_preds = 0
    local n_correct = 0
    local n_shards = opt.n_te_shards
    local str_split = 'te_'
    if split == "val" then
        n_shards = opt.n_val_shards
        str_split = 'val_'
    end

    local preds = torch.zeros(opt.batch_size*opt.kB)

    for shard_n = 1, n_shards do 
        local f = hdf5.open(opt.data_folder .. str_split .. shard_n .. '.hdf5', 'r')
        local sp_ins = f:read('ins'):all()
        local sp_outs = f:read('outs'):all()
        local sp_data = Data(opt, {sp_ins, sp_outs})

        for i = 1, sp_data.n_batches do
            local episode = sp_data[i]
            inputs, targs = episode[1], episode[2]
            local bat_embs = torch.reshape(model:forward(inputs[1]), opt.batch_size, opt.kB, opt.n_classes)
            local set_embs = torch.reshape(model:forward(inputs[2]), opt.batch_size, opt.N*opt.k, opt.n_classes)
            local sim_scores = torch.zeros(opt.batch_size, opt.kB, opt.N*opt.k)
            if opt.gpuid > 0 then
                sim_scores = sim_scores:cuda()
            end
            sim_scores:baddbmm(bat_embs, set_embs:transpose(2,3))
            local unbatch = torch.reshape(sim_scores, opt.batch_size*opt.kB, opt.N*opt.k)
            local maxes, inds = torch.max(unbatch, 2)
            for i=1, inputs[3]:size(1) do -- should probably assert sizes
                for j=1,opt.kB do
                    preds[(i-1)*opt.kB+j] = inputs[3][i][inds[(i-1)*opt.kB+j][1]]
                end
            end
                
            if opt.gpuid > 0 then
                preds = preds:cuda()
            end
            n_correct = n_correct + torch.sum(torch.eq(preds, targs))
            n_preds = n_preds + targs:nElement()
            
            if flag == 1 and opt.debug == 1 then
                dbg()
            end

            if fh ~= nil then
                for j = 1, preds:nElement() do
                    fh:write(preds[j][1] .. ', ' .. targs[j] .. "\n")
                end
            end
        end
        sp_ins = nil
        sp_outs = nil
        sp_data = nil
        collectgarbage()
    end
    return n_correct/n_preds
end
