package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'

--
-- Various models used throughout experiments
--

function make_fce_g(opt)
    local inputs = {nn.Identity()()} -- set embs; B x n_set x d_emb
    local hs = nn.SeqBRNN(opt.d_emb, opt.d_emb, true)(inputs[1])
    local gs = nn.CAddTable()({hs, inputs[1]})
    local outputs = {gs}
    return nn.gModule(inputs, outputs)
end

function make_simple_fce(opt)
    local inputs = {}
    local d_emb, kB, n_set = opt.d_emb, opt.kB, opt.n_set
    table.insert(inputs, nn.Identity()()) -- test embs; B x kb x d_emb
    table.insert(inputs, nn.Identity()()) -- set embs; B x n_set x d_emb
    
    unbatched_bat = nn.View(-1, d_emb)(inputs[1])
    repeat_bat = nn.View(-1, d_emb)(
        nn.Contiguous()(nn.Replicate(n_set, 2)(unbatched_bat)))

    repeated_set = nn.Replicate(kB, 1)(inputs[2])
    unbatched_set = nn.View(-1, d_emb)(nn.Contiguous()(repeated_set))

    concat = nn.JoinTable(2)({repeat_bat, unbatched_set})
    w1 = nn.Linear(2*d_emb, d_emb)(concat)
    nonlinear = nn.ReLU()(w1)
    w2 = nn.Linear(d_emb, d_emb)(nonlinear)
    unbatched = nn.View(-1, n_set, d_emb)(w2)

    local outputs = {}
    table.insert(outputs) -- test embs: (B*kB) x 1 x d
    table.insert(outputs, unbatched) -- set embs: (B*kB) x n_set x d
    return nn.gModule(inputs, outputs)
end

function make_fce_f(opt, n_set)
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- test embs; B x kB x d_emb
    table.insert(inputs, nn.Identity()()) -- set embs; B x n_set x d_emb
    table.insert(inputs, nn.Identity()()) -- c_0; (B*kB) x d_emb
    table.insert(inputs, nn.Identity()()) -- h_0; (B*kB) x d_emb

    -- (B*kB) x d_emb
    local reshaped_x = nn.View(-1, opt.d_emb)(inputs[1])

    local prev_c, prev_h = inputs[3], inputs[4]
    local prev_h_hat
    for l = 1, opt.L do -- L is the number of processing steps
        -- TODO dropout?

        -- reshape_h: B x kB x d_emb
        -- attn_scores: (B*kB) x n_set
        -- attn_vec = (B*kB) x d_emb
        --   Unsqueeze: (B*kB) x n_set x 1
        --   Replicate: B x kB x n_set x d_emb
        --     Transpose: kB x B x n_set x d_emb
        --     Reshape: (B*kB) x n_set x d_emb
        -- next_hid: (B*kB) x 2*d_emb
        local reshape_h = nn.View(-1, opt.kB, opt.d_emb)(prev_h)
        local attn_scores = nn.SoftMax()(nn.View(-1, n_set)(
            nn.MM(false, true)({
            reshape_h, inputs[2]})))
        local attn_vec = nn.MM({true, false})({
            nn.Reshape(opt.kB*opt.batch_size, n_set, opt.d_emb)(
            nn.Transpose({1,2})(
            nn.Replicate(opt.kB, 1, 2)(inputs[2]))),
            nn.Unsqueeze(3)(attn_scores)})
        concat_hid = nn.JoinTable(2)({
            prev_h, attn_vec})

        -- do all the linear transforms at once
        -- out: (B*kB) x 4*d_emb
        local i2h = nn.Linear(opt.d_emb, 4*opt.d_emb)(reshaped_x)
        local h2h = nn.Linear(2*opt.d_emb, 4*opt.d_emb, false)(
            concat_hid)
        local linear = nn.CAddTable()({i2h, h2h})

        -- split into the four different linear layers
        -- Reshape: (B*kB) x 4 x d_emb
        -- SplitTable: 4 x (B*kB) x d_emb
        local reshaped_linear = nn.Reshape(4, opt.d_emb)(linear) 
        local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped_linear):split(4)
        local in_gate = nn.Sigmoid()(n1)
        local forget_gate = nn.Sigmoid()(n2)
        local out_gate = nn.Sigmoid()(n3)
        local in_mlp = nn.Tanh()(n4)

        -- compute next h, c
        -- next_c: (B*kB) x d_emb
        -- next_h: (B*kB) x d_emb
        prev_c = nn.CAddTable()({
            nn.CMulTable()({forget_gate, prev_c}),
            nn.CMulTable()({in_gate, in_mlp})})
        prev_h_hat = nn.CMulTable()({
            out_gate, nn.Tanh()(prev_c)})
        prev_h = nn.CAddTable()({
            reshaped_x, prev_h_hat})

        --[[
        if l <= opt.L then
            -- reshape_h: B x kB x d_emb
            -- attn_scores: (B*kB) x n_set
            -- attn_vec = (B*kB) x d_emb
            --   Unsqueeze: (B*kB) x n_set x 1
            --   Replicate: B x kB x n_set x d_emb
            --     Transpose: kB x B x n_set x d_emb
            --     Reshape: (B*kB) x n_set x d_emb
            -- next_hid: (B*kB) x 2*d_emb
            local reshape_h = nn.View(-1, opt.kB, opt.d_emb)(next_h)
            local attn_scores = nn.SoftMax()(nn.View(-1, n_set)(
                nn.MM(false, true)({
                reshape_h, inputs[2]}))) -- inputs[2]: B x n_set x d_emb
            local attn_vec = nn.MM({true, false})({
                nn.Reshape(opt.kB*opt.batch_size, n_set, opt.d_emb)(
                nn.Transpose({1,2})(
                nn.Replicate(opt.kB, 1, 2)(inputs[2]))),
                nn.Unsqueeze(3)(attn_scores)})
            next_hid = nn.JoinTable(2)({
                next_h, attn_vec})
        end
        --]]
    end
    local outputs = {nn.Reshape(opt.batch_size, opt.kB, opt.d_emb)(prev_h)}
    return nn.gModule(inputs, outputs)
end

function make_bow(opt)
    local inputs = {nn.Identity()()} -- B*kB x seq_len
    local embs = nn.LookupTable(opt.vocab_size, opt.d_emb)(inputs[1])
    if opt.pretrain_file ~= '' then
        local f = hdf5.open(opt.data_folder .. 'embs.hdf5', 'r')
        local weights = f:read('embs'):all()
        assert(weights:size()[2] == opt.d_emb)
        embs.weight = weights
    end
    local BoWs = nn.Sum(2)(embs) -- Mean doesn't account for blanks
    local outputs = {BoWs} -- B*kB x d_emb
    return nn.gModule(inputs, outputs), embs.weight
end

function make_lstm(opt)
    local inputs = {nn.Identity()()} -- B*kB x seq_len
    local embs = nn.LookupTable(opt.vocab_size, opt.d_emb)(inputs[1])
    if opt.pretrain_file ~= '' then
        local f = hdf5.open(opt.data_folder .. 'embs.hdf5', 'r')
        local weights = f:read('embs'):all()
        embs.weight = weights
    end
    local lstm = nn.SeqLSTM(opt.d_emb, opt.d_emb)
    lstm.batchfirst = true
    local sent_emb = nn.SelectTable(-1)(nn.SplitTable(2)(lstm(embs)))
    local outputs = {sent_emb} -- B*kb x d_emb
    return nn.gModule(inputs, outputs), embs.weight
end

function make_cnn(opt)
    local input = {nn.Identity()()}
    local layers = {}
    local kernel_sizes = (opt.kernel_sizes):split(',')
    assert(#kernel_sizes == opt.n_modules)
    for i = 1, #kernel_sizes do
        kernel_sizes[i] = tonumber(kernel_sizes[i])
    end
    for i=1,opt.n_modules do
        if i == 1 then
            local conv_module = make_cnn_module(opt, opt.n_channels, kernel_sizes[i])
            conv_module.name = 'cnn' .. i
            table.insert(layers, conv_module(input))
        else
            local conv_module = make_cnn_module(opt, opt.d_emb, kernel_sizes[i])
            conv_module.name = 'cnn' .. i
            table.insert(layers, conv_module(layers[i-1]))
        end
    end
    local output = {nn.Squeeze()(layers[opt.n_modules])}
    return nn.gModule(input, output)
end

function make_cnn_module(opt, n_input_feats, kernel_size)
    local conv_w = kernel_size
    local conv_h = kernel_size
    local pad_w = math.floor((conv_w - 1) / 2)
    local pad_h = math.floor((conv_h - 1) / 2)
    local pool_w = opt.pool_width
    local pool_h = opt.pool_height

    local nonlinearity
    if opt.nonlinearity == 'relu' then
        nonlinearity = nn.ReLU()
    elseif opt.nonlinearity == 'tanh' then
        nonlinearity = nn.Tanh()
    end

    local input = {nn.Identity()()}
    local output = {}
    if opt.gpuid > 0 and opt.cudnn == 1 then
        -- 1 is the number of input channels since b&w
        -- conv height and width change every layer
        local conv_layer = cudnn.SpatialConvolution(n_input_feats, opt.d_emb, 
            conv_w, conv_h, 1, 1, pad_w, pad_h)(input)
        local norm_layer = nn.SpatialBatchNormalization(opt.d_emb, opt.bn_eps, opt.bn_momentum, opt.bn_affine)(conv_layer)
        local pool_layer = cudnn.SpatialMaxPooling(
            pool_w, pool_h, pool_w, pool_h, opt.pool_pad, opt.pool_pad)
        if opt.pool_ceil == 1 then
            pool_layer:ceil()
        end
        table.insert(output, pool_layer(nn.Tanh()(norm_layer)))
    else
        local conv_layer = nn.SpatialConvolution(n_input_feats, opt.d_emb,
            conv_w, conv_h, 1, 1, pad_w, pad_h)(input)
        local norm_layer = nn.SpatialBatchNormalization(opt.d_emb, opt.bn_eps, opt.bn_momentum, opt.bn_affine)(conv_layer)
        local pool_layer = nn.SpatialMaxPooling(
            pool_w, pool_h, pool_w, pool_h, opt.pool_pad, opt.pool_pad)
        if opt.pool_ceil == 1 then
            pool_layer:ceil()
        end
        table.insert(output, pool_layer(nn.ReLU()(norm_layer)))
    end
    return nn.gModule(input, output)
end
