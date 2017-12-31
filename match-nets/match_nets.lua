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
    table.insert(inputs, nn.Identity()()) -- hat(x): B*kb x ...
    table.insert(inputs, nn.Identity()()) -- x_i: B*N*k x ...
    table.insert(inputs, nn.Identity()()) -- y_i: B x N*k
    if opt.contextual_f == 'fce' then
        table.insert(inputs, nn.Identity()()) -- h_0
        table.insert(inputs, nn.Identity()()) -- c_0
    end

    -- in: B*kB x ... (depends on input)
    --   f : B*kB x d_emb
    --   g : B*n_set x d_emb
    --   normalize : B*{kB,n_set} x d_emb
    --   view : B x {kB,n_set} x d_emb
    -- out: B x kB x n_kern
    local n_set = opt.N*opt.k
    local f, g
    if opt.embedding_fn == 'cnn' then
        f = make_cnn(opt)
        g = make_cnn(opt)
    elseif opt.embedding_fn == 'bow' then
        f, self.word_embs = make_bow(opt)
        g, self.word_embs = make_bow(opt)
    elseif opt.embedding_fn == 'lstm' then
        f, self.word_embs = make_lstm(opt)
        g, self.word_embs = make_lstm(opt)
    else
        f = nn.View(-1, opt.d_emb)
        g = nn.View(-1, opt.d_emb)
    end
    if opt.share_embed == 1 then
        log(log_fh, '\tTying embedding function parameters...')
        g:share(f, 'weight')
        g:share(f, 'bias')
        g:share(f, 'gradWeight')
        g:share(f, 'gradBias')
    end
    local embed_f = f(inputs[1])
    local norm_f = nn.Normalize(2)(embed_f)
    local batch_f = nn.View(-1, opt.kB, opt.d_emb)(norm_f)
    local embed_g = g(inputs[2])
    local norm_g = nn.Normalize(2)(embed_g)
    local batch_g = nn.View(-1, n_set, opt.d_emb)(norm_g)

    -- Class prototypes
    -- in: B x n_set x n_kern
    --   IndexAdd: B x N x n_kern (we don't divide by k b/c we normalize later)
    --   View: B*N x n_kern
    --   Normalize: B*N x n_kern
    --   View: B x N x n_kern
    -- out: B x n_set x n_kern
    if opt.prototypes == 1 then
        log(log_fh, '\tUsing prototypes...')
        n_set = opt.N
        prototypes = nn.IndexAdd(1, opt.N)({batch_g, inputs[3]})
        batch_g = nn.View(-1, n_set, opt.d_emb)(
            nn.Normalize(2)(nn.View(-1, opt.d_emb)(prototypes)))
    end
    opt.n_set = n_set

    -- Contextual embeddings: parameters after embedding
    -- in:(B x kB x n_kern) , (B x n_set x n_kern)
    -- out: (B x kB x n_kern), (B x n_set x n_kern)
    --[[
    if opt.contextual_embed == 'fce' then
        log(log_fh, '\tUsing full context embeddings...')
        local fce_f, fce_g, h_0, c_0
        if opt.contextual_f == 1 then
            log(log_fh, '\t\tfor f...')
            if opt.init_fce == 'zero' then
                h_0 = torch.zeros(opt.batch_size*opt.kB, 2*opt.d_emb)
                c_0 = torch.zeros(opt.batch_size*opt.kB, opt.d_emb)
            else
                h_0 = torch.rand(2*opt.d_emb) - .5 * 2*opt.init_scale
                c_0 = torch.rand(opt.d_emb) -.5 * 2*opt.init_scale
            end
            if opt.gpuid > 0 then
                h_0 = h_0:cuda()
                c_0 = c_0:cuda()
            end
            self.h_0 = h_0
            self.c_0 = c_0

            fce_f = make_fce_f(opt, n_set)
            batch_f = fce_f({batch_f, batch_g, inputs[4], inputs[5]})
        end
        if opt.contextual_g == 1 then
            log(log_fh, '\t\tfor g...')
            fce_g = make_fce_g(opt)
            batch_g = fce_g({batch_g})
        end
    elseif opt.contextual_embed == 'simple' then
        log(log_fh, '\tUsing simple contextual embeddings...')
        simple_fce = make_simple_fce(opt)
        contextual_embeds = simple_fce({batch_f, batch_g})
        batch_f = nn.SplitTable(1)(contextual_embeds)
        batch_g = nn.SplitTable(2)(contextual_embeds)
    end
    --]]

    local fce_f, fce_g, h_0, c_0
    if opt.contextual_f == 'fce' then
        log(log_fh, '\tUsing full context embeddings for f...')
        if opt.init_fce == 'zero' then
            h_0 = torch.zeros(opt.batch_size*opt.kB, opt.d_emb)
            c_0 = torch.zeros(opt.batch_size*opt.kB, opt.d_emb)
        else
            h_0 = torch.rand(opt.d_emb) - .5 * 2*opt.init_scale
            c_0 = torch.rand(opt.d_emb) - .5 * 2*opt.init_scale
        end
        if opt.gpuid > 0 then
            h_0 = h_0:cuda()
            c_0 = c_0:cuda()
        end
        self.h_0 = h_0
        self.c_0 = c_0

        fce_f = make_fce_f(opt, n_set)
        batch_f = fce_f({batch_f, batch_g, inputs[4], inputs[5]})
    end

    if opt.contextual_g == 'simple' then
        log(log_fh, '\tUsing simple contextual embeddings for g...')
        fce_g = make_simple_fce_g(opt)
        batch_g = fce_g({batch_f, batch_g})
    elseif opt.contextual_g == 'fce' then
        log(log_fh, '\tUsing full context embeddings for g...')
        fce_g = make_fce_g(opt)
        batch_g = fce_g({batch_g})
    end

    -- in:(B x kB x n_kern) , (B x n_set x n_kern)
    --   MM:: B x n_set x kB
    --   Transpose: B x kB x n_set
    --   View: (B*kB) x n_set
    --   SoftMax: (B*kB) x n_set
    --   (View): B x kB x n_set 
    --   (Transpose): B x n_set x kB
    --   (IndexAdd): B x N x kB
    --   (Transpose): B x kB x N
    --   (View): (B*kB) x N
    -- out: (B*kB) x N
    local cos_dist = nn.MM(false, true)({batch_g, batch_f})
    local unbatch = nn.View(-1, n_set)(nn.Transpose({2,3})(cos_dist))
    local set_probs = nn.SoftMax()(unbatch)
    local class_probs = set_probs
    if opt.prototypes ~= 1 then
        local rebatch = nn.Transpose({2,3})(
            nn.View(-1, opt.kB, n_set)(class_probs))
        class_probs = nn.View(-1, opt.N)(nn.Transpose({2,3})(
            nn.IndexAdd(1, opt.N)({rebatch, inputs[3]})))
    end
    local log_probs = nn.Log()(class_probs)
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
end

function MatchingNetwork:train(log_fh)
    local model = self.model
    local crit = self.crit
    local opt = self.opt

    local debug_pts = (opt.debug_on):split(',')
    for i = 1, #debug_pts do
        debug_pts[i] = tonumber(debug_pts[i])
    end

    --[[ Parameter Init ]]--
    local params, grad_params = model:getParameters()
    if opt.init_dist == 'normal' then
        params:normal(0, opt.init_scale)
    elseif opt.init_dist == 'uniform' then
        params:uniform(-opt.init_scale, opt.init_scale)
    elseif opt.init_dist == 'default' then
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
        log(log_fh, "\t\twith weight decay " .. opt.weight_decay)
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

    if opt.debug == 1 then
        for pt = 1, #debug_pts do
            if debug_pts[pt] == 0 then
                dbg()
            end
        end
    end


    --[[ Training Loop ]]--
    local timer = torch.Timer()
    local last_score = self:evaluate("val")
    local best_score = last_score
    log(log_fh, "\tInitial validation accuracy: " .. last_score)
    if opt.save_model_to ~= '' then
        timer:reset()
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
            local tr_data = Data(opt, {tr_ins, tr_outs})
            for i = 1, tr_data.n_batches do
                if opt.embedding_fn == 'bow' or opt.embedding_fn == 'lstm' then
                    self.word_embs[1]:zero() -- corresponds to <blank>
                end
                local episode = tr_data[i]
                inputs, targs = episode[1], episode[2]
                if opt.contextual_f == 'fce' then
                    table.insert(inputs, self.c_0)
                    table.insert(inputs, self.h_0)
                end
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

        val_score = self:evaluate("val")
        log(log_fh, "\tValidation time " .. timer:time().real .. " seconds")
        if opt.learning_rate_decay > 0 and opt.optimizer == 'adagrad' and val_score <= last_score then
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
            log(log_fh,"\t\tBEST PARAMETERS!!!")
        end
        last_score = val_score
        log(log_fh, "\tLoss: " .. total_loss/n_batches .. ", training accuracy: " .. n_correct/n_preds .. ", Validation accuracy: " .. val_score .. ", Best accuracy: " .. best_score)

        if opt.debug == 1 then
            for pt = 1, #debug_pts do
                if debug_pts[pt] == epoch then
                    dbg()
                end
            end
        end

    end
end

function MatchingNetwork:evaluate(split)
    local model = self.model
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
    for shard_n = 1, n_shards do 
        local f = hdf5.open(opt.data_folder .. str_split .. shard_n .. '.hdf5', 'r')
        local sp_ins = f:read('ins'):all()
        local sp_outs = f:read('outs'):all()
        local sp_data = Data(opt, {sp_ins, sp_outs})

        for i = 1, sp_data.n_batches do
            local episode = sp_data[i]
            inputs, targs = episode[1], episode[2]
            if opt.contextual_f == 'fce' then
                table.insert(inputs, self.c_0)
                table.insert(inputs, self.h_0)
            end
            local outputs = model:forward(inputs)
            maxes, preds = torch.max(outputs, 2)
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
