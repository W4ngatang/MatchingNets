package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'

local MatchingNetwork = torch.class("MatchingNetwork")

--
-- Various models used throughout experiments
--

function MatchingNetwork:__init(opt, log_fh)
    -- input is support set and labels, test datum
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- hat(x): B*kb x im x im
    table.insert(inputs, nn.Identity()()) -- x_i: B*N*k x im x im
    table.insert(inputs, nn.Identity()()) -- y_i: B x N*k

    -- in: B*kB x im x im
    --   unsqueeze -> B*kB x 1 x im x im
    --   f -> B*kB x 64 x 1 x 1
    --   squeeze -> B*kB x 64
    --   normalize -> B*kB x 64
    --   reshape -> B x kB x 64
    -- out: B x kB x 64
    local f = make_cnn(opt)
    local embed_f = nn.Squeeze()(f(nn.Unsqueeze(2)(inputs[1])))
    local norm_f = nn.Normalize(2)(embed_f)
    local batch_f = nn.View(-1, opt.kB, opt.n_kernels)(norm_f)

    -- in: B*N*k x im x im
    --   unsqueeze -> B*N*k x 1 x im x im
    --   g -> B*N*k x 64 x 1 x 1
    --   squeeze -> B*N*k x 64
    --   normalize -> B*N*k x 64
    --   view -> B x N*k x 64
    -- out: B x N*k x 64
    local g = make_cnn(opt)
    if opt.share_embed == 1 then
        log(log_fh, '\tTying embedding function parameters...')
        g:share(f, 'weight')
        g:share(f, 'bias')
        g:share(f, 'gradWeight')
        g:share(f, 'gradBias')
    end
    local embed_g = nn.Squeeze()(g(nn.Unsqueeze(2)(inputs[2])))
    local norm_g = nn.Normalize(2)(embed_g)
    local batch_g = nn.View(-1, opt.N*opt.k, opt.n_kernels)(norm_g)
    
    -- in (B x kB x 64) , (B x N*k x 64)
    --   MM: -> B x N*k x kB
    --   Transpose: B x kB x N*k
    --   View -> (B * kB) x N*k
    --   SoftMax -> (B * kB) x N*k
    --   View -> B x kB x N*k
    --   Transpose -> B x N*k x kB
    --   IndexAdd -> B x N x kB
    --   Transpose -> B x kB x N
    --   View -> B*kB x N
    local cos_dist = nn.MM(false, true)({batch_g, batch_f})
    local unbatch1 = nn.View(-1, opt.N*opt.k)(nn.Transpose({2,3})(cos_dist))
    local attn_scores = nn.SoftMax()(unbatch1)
    local rebatch = nn.Transpose({2,3})(nn.View(-1, opt.kB, opt.N*opt.k)(attn_scores))
    local class_probs = nn.IndexAdd(1, opt.N)({rebatch, inputs[3]})
    local unbatch2 = nn.View(-1, opt.N)(nn.Transpose({2,3})(class_probs))
    local log_probs = nn.Log()(unbatch2)
    local outputs = {log_probs}
    local crit = nn.ClassNLLCriterion()
    local model = nn.gModule(inputs, outputs)
    if opt.gpuid > 0 then
        model = model:cuda()
        crit = crit:cuda()
    end
    self.model = model
    self.crit = crit
    self.opt = opt
    self.log_fh = log_fh
    self.flag = 0
end

function MatchingNetwork:train()
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
        --params = params + .01
    else
        log(log_fh, 'Unsupported distribution for initialization!')
        return
    end

    --[[ Optimization Setup ]]--
    log(log_fh, "\tOptimizing with " .. opt.optimizer)
    if opt.optimizer == 'sgd' then
        optim_state = { -- NB: Siamese paper used layerwise LR and momentum
            learningRate = opt.learning_rate,
            weightDecay = opt.weight_decay,
            momentum = opt.momentum,
            learningRateDecay = opt.learning_rate_decay
        }
        optimize = optim.sgd
        log(log_fh, "\t\twith learning rate " .. opt.learning_rate)
        log(log_fh, "\t\twith learning rate decay " .. opt.learning_rate_decay)
        log(log_fh, "\t\twith weight decay" .. opt.weight_decay)
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
        log(log_fh, "\t\twith learning rate decay " .. opt.learning_rate_decay)
        log(log_fh, "\t\twith beta1 " .. opt.beta1)
        log(log_fh, "\t\twith beta2 " .. opt.beta2)
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
            local tr_data = Data(opt, {tr_ins, tr_outs})
            for i = 1, tr_data.n_batches do
                local episode = tr_data[i]
                inputs, targs = episode[1], episode[2]
                optimize(feval, params, optim_state)
            end
            if shard_n % (10/opt.print_freq) == 0 then
                log(log_fh, "\t  Completed " .. shard_n/opt.n_tr_shards*100 .. "% in " ..timer:time().real .. " seconds")
            end
            n_batches = n_batches + tr_data.n_batches
            tr_ins = nil
            tr_outs = nil
            tr_data = nil
            collectgarbage()
        end
        log(log_fh, "\tTraining time: " .. timer:time().real .. " seconds")

        if epoch >= opt.debug_after and opt.debg == 1 then
            flag = 1
        end
        if opt.debug == 1 and flag == 1 then
            dbg()
        end

        timer:reset()
        val_score = self:evaluate("val")
        log(log_fh, "\tValidation time " .. timer:time().real .. " seconds")
        if opt.learning_rate_decay > 0 and opt.optimizer == 'adagrad' and val_score <= last_score then
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

function MatchingNetwork:evaluate(split)
    local model = self.model
    local crit = self.crit
    local opt = self.opt
    local log_fh = self.log_fh

    if opt.predfile ~= nil and split == 'te' then
        pred_fh = io.open(opt.predfile,"w")
        pred_fh:write("Prediction, Target\n")
    end

    model:evaluate()
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
        local sp_data = Data(opt, {sp_ins, sp_outs})

        for i = 1, sp_data.n_batches do
            local episode = sp_data[i]
            inputs, targs = episode[1], episode[2]
            local outputs = model:forward(inputs)
            maxes, preds = torch.max(outputs, 2)
            if opt.gpuid > 0 then
                preds = preds:cuda()
            end
            n_correct = n_correct + torch.sum(torch.eq(preds, targs))
            n_preds = n_preds + targs:nElement()
            
            --[[
            if flag == 1 and opt.debug == 1 then
                dbg()
            end
            ]]--

            if pred_fh ~= nil then
                for j = 1, preds:nElement() do
                    pred_fh:write(preds[j][1] .. ', ' .. targs[j] .. "\n")
                end
            end
        end
        sp_ins = nil
        sp_outs = nil
        sp_data = nil
        collectgarbage()
    end
    if pred_fh ~= nil then
        pred_fh:close()
    end
    return n_correct/n_preds
end