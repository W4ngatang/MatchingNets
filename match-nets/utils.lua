package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'

--
-- Various models used throughout experiments
--

function make_fce_g(opt)
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- set embs; B x n_set x n_kern
    local brnn = make_brnn(opt)

    local gs = nn.CAddTable()({gs_hat, inputs[1]})
    local outputs = {gs}
    return nn.gModule({inputs, outputs})
end

function make_brnn(opt)
    local fwd = nn.FastLSTM(opt.n_kernels, opt.n_kernels)
    local bwd = fwd:clone()
    bwd:reset()
    local merge = nn.CAddTable() -- will add the two hidden states
    local brnn = nn.BiSequencer(fwd)
    local rnn = nn.Sequential():add(brnn)

    return rnn
end

function make_fce_f(opt, n_set)
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- test embs; B x kB x n_kern
    table.insert(inputs, nn.Identity()()) -- set embs; B x n_set x n_kern
    table.insert(inputs, nn.Identity()()) -- c_0; (B*kB) x n_kern
    table.insert(inputs, nn.Identity()()) -- h_0; (B*kB) x (2*n_kern) 
        -- TODO needs to be (B*kB) x n_kern

    -- (B*kB) x n_kern
    local reshaped_x = nn.View(-1, opt.n_kernels)(inputs[1])

    local prev_c, prev_hid, next_c, next_hid
    for l = 1, opt.L do -- L is the number of processing steps
        if l == 1 then
            prev_c = inputs[3]
            prev_hid = inputs[4]
        else
            prev_c = next_c
            prev_hid = next_hid
        end

        -- TODO dropout?

        -- do all the linear transforms at once
        -- out: (B*kB) x 4*n_kern
        local i2h = nn.Linear(opt.n_kernels, 4*opt.n_kernels)(reshaped_x)
        local h2h = nn.Linear(2*opt.n_kernels, 4*opt.n_kernels, false)(
            prev_hid)
        local linear = nn.CAddTable()({i2h, h2h})

        -- split into the four different linear layers
        -- Reshape: (B*kB) x 4 x n_kern
        -- SplitTable: 4 x (B*kB) x n_kern TODO how to split / unpack it?
        local reshaped_linear = nn.Reshape(4, opt.n_kernels)(linear) 
        local split_linear = nn.SplitTable(2)(reshaped)
        local in_gate = nn.Sigmoid()(split_linear[1])
        local forget_gate = nn.Sigmoid()(split_linear[2])
        local out_gate = nn.Sigmoid()(split_linear[3])
        local in_mlp = nn.Tanh()(split_linear[4])

        -- compute next h, c
        -- next_c: (B*kB) x n_kern
        -- next_h: (B*kB) x n_kern
        next_c = nn.CAddTable()({
            nn.CMulTable()({forget_gate, prev_c}),
            nn.CMulTable()({in_gate, in_mlp})})
        local next_h_hat = nn.CMulTable()({
            out_gate, nn.Tanh()(next_c)})
        local next_h = nn.CAddTable()({
            reshaped_x, next_h_hat})

        -- reshape_h: B x kB x n_kern
        -- attn_scores: (B*kB) x n_set
        -- attn_vec = (B*kB) x n_kern
        -- next_hid: (B*kB) x 2*n_kern
        local reshape_h = nn.View(-1, opt.kB, opt.n_kernels)(next_h)
        local attn_scores = nn.SoftMax()(nn.View(-1, n_set)(
            nn.MM({false, true})({
            next_h_copies, inputs[2]})))
        local attn_vec = nn.MM({true, false})({
            nn.Unsqueeze(3)(attn_scores), 
            nn.Reshape(opt.kB*opt.batch_size, opt.n_set, opt.n_kernels)(
            nn.Transpose({1,2})(nn.Replicate(opt.kB, 1, 2)(inputs[2])))})
        next_hid = nn.JoinTable(2)({
            next_h, attn_vec})
    end
    local outputs = {next_hid}
    return nn.gModule(inputs, outputs)
end

function make_cnn(opt)
    local input = nn.Identity()()
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
            local conv_module = make_cnn_module(opt, opt.n_kernels, kernel_sizes[i])
            conv_module.name = 'cnn' .. i
            table.insert(layers, conv_module(layers[i-1]))
        end
    end
    local output = layers[opt.n_modules]
    return nn.gModule({input}, {output})
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

    local output
    local input = nn.Identity()() -- double () denotes nngraph module
    if opt.gpuid > 0 and opt.cudnn == 1 then
        -- 1 is the number of input channels since b&w
        -- conv height and width change every layer
        local conv_layer = cudnn.SpatialConvolution(n_input_feats, opt.n_kernels, 
            conv_w, conv_h, 1, 1, pad_w, pad_h)(input)
        local norm_layer = nn.SpatialBatchNormalization(opt.n_kernels, opt.bn_eps, opt.bn_momentum, opt.bn_affine)(conv_layer)
        local pool_layer = cudnn.SpatialMaxPooling(
            pool_w, pool_h, pool_w, pool_h, opt.pool_pad, opt.pool_pad)
        if opt.pool_ceil == 1 then
            pool_layer:ceil()
        end
        output = pool_layer(nn.Tanh()(norm_layer))
    else
        local conv_layer = nn.SpatialConvolution(n_input_feats, opt.n_kernels,
            conv_w, conv_h, 1, 1, pad_w, pad_h)(input)
        local norm_layer = nn.SpatialBatchNormalization(opt.n_kernels, opt.bn_eps, opt.bn_momentum, opt.bn_affine)(conv_layer)
        local pool_layer = nn.SpatialMaxPooling(
            pool_w, pool_h, pool_w, pool_h, opt.pool_pad, opt.pool_pad)
        if opt.pool_ceil == 1 then
            pool_layer:ceil()
        end
        output = pool_layer(nn.ReLU()(norm_layer))
    end
    return nn.gModule({input},{output})
end
