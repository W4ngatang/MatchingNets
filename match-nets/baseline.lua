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
    --   f -> B x d_emb
    --   linear -> B x n_classes
    --   tanh -> B x n_classes
    -- out: B x n_classes
    local f
    if opt.embedding_fn == 'cnn' then
        f = make_cnn(opt)
    elseif opt.embedding_fn == 'bow' then
        f, self.word_embs = make_bow(opt)
    elseif opt.embedding_fn == 'lstm' then
        f, self.word_embs = make_lstm(opt)
    else
        f = nn.View(-1, opt.d_emb)
    end
    local embed_f = f(inputs[1])
    local norm_f = nn.Normalize(2)(embed_f)
    local linear = nn.Linear(opt.d_emb, opt.n_classes, false)(norm_f)
    local activation = nn.Tanh()(linear)
    local outputs = {activation}
    local crit = nn.CrossEntropyCriterion()
    local model = nn.gModule(inputs, outputs)
    if opt.gpuid > 0 then
        model = model:cuda()
        crit = crit:cuda()
    end
    self.embed = f --nn.gModule(inputs, {embed_f})
    self.model = model
    self.crit = crit
    self.opt = opt
end

function Baseline:train(log_fh)
    local model = self.model
    local crit = self.crit
    local opt = self.opt

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
    log(log_fh, "\tInitial validation accuracy: " .. last_score)
    if opt.save_model_to ~= '' then
        torch.save(opt.save_model_to, self)
        log(log_fh, "\tSaved model to " .. opt.save_model_to .. " in ".. timer:time().real .. " seconds")
    end

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
                if opt.embedding_fn == 'bow' or opt.embedding_fn == 'lstm' then
                    self.word_embs[1]:zero()
                end
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
            if opt.save_model_to ~= '' then
                timer:reset()
                torch.save(opt.save_model_to, self)
                log(log_fh, "\tSaved model to " .. opt.save_model_to .. " in ".. timer:time().real .. " seconds")
            end
            best_score = val_score
        end
        last_score = val_score
        log(log_fh, "\tLoss: " .. total_loss/n_batches .. ", training accuracy: " .. n_correct/n_preds .. ", Validation accuracy: " .. val_score .. ", Best accuracy: " .. best_score)
    end

end

function Baseline:evaluate(split, fh)
    local model = self.embed --self.model
    local crit = self.crit
    local opt = self.opt

    if opt.predfile ~= '' and split == 'te' then
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

    local preds = torch.zeros(opt.batch_size*opt.kB)
    local softmax = nn.SoftMax()
    if opt.gpuid > 0 then
        softmax = softmax:cuda()
    end

    for shard_n = 1, n_shards do 
        local f = hdf5.open(opt.data_folder .. str_split .. shard_n .. '.hdf5', 'r')
        local sp_ins = f:read('ins'):all()
        local sp_outs = f:read('outs'):all()
        local sp_data = Data(opt, {sp_ins, sp_outs})

        for i = 1, sp_data.n_batches do
            local episode = sp_data[i]
            inputs, targs = episode[1], episode[2]
            local bat_embs = torch.reshape(torch.renorm(model:forward(inputs[1]), 2,1,1), opt.batch_size, opt.kB, opt.d_emb) -- B x kB x n_classes
            local set_embs = torch.reshape(torch.renorm(model:forward(inputs[2]), 2,1,1),opt.batch_size, opt.N*opt.k, opt.d_emb) -- B x n_set x n_classes
            -- B x kB x n_set
            local logits = torch.zeros(opt.batch_size, opt.kB, opt.N*opt.k)
            local class_probs = torch.zeros(opt.batch_size, opt.kB, opt.N)
            if opt.gpuid > 0 then
                logits = logits:cuda()
                class_probs = class_probs:cuda()
            end
            logits:baddbmm(bat_embs, set_embs:transpose(2,3))
            -- B x kB x n_set
            local unbatch = torch.reshape(logits, opt.batch_size*opt.kB, opt.N*opt.k)
            local probs = softmax:forward(unbatch)
            local rebatch = torch.reshape(probs, opt.batch_size, opt.kB, opt.N*opt.k)
            for j = 1, opt.batch_size do
                class_probs[j]:indexAdd(2, inputs[3][j], rebatch[j])
            end
            local maxes, preds = torch.max(
                torch.reshape(class_probs, opt.batch_size*opt.kB, opt.N), 2)

            --[[
            local maxes, inds = torch.max(unbatch, 2)
            for a=1, inputs[3]:size(1) do -- should probably assert sizes
                for b=1,opt.kB do
                    preds[(a-1)*opt.kB+b] = inputs[3][a][inds[(a-1)*opt.kB+b][1]
                end
            end
            ]]--
                
            if opt.gpuid > 0 then
                preds = preds:cuda()
            end
            n_correct = n_correct + torch.sum(torch.eq(preds, targs))
            n_preds = n_preds + targs:nElement()
            
            if flag == 1 and opt.debug == 1 then
                dbg()
            end

            if pred_fh ~= nil then
                for j = 1, preds:nElement() do
                    pred_fh:write(preds[j] .. ', ' .. targs[j] .. "\n")
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
