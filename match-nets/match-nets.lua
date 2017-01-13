package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'

--
-- Various models used throughout experiments
--

function make_matching_net(opt)
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
        g:share(f, 'weight')
        g:share(f, 'bias')
        g:share(f, 'gradWeight')
        g:share(f, 'gradBias')
    end
    local embed_g = nn.Squeeze()(g(nn.Unsqueeze(2)(inputs[2])))
    local norm_g = nn.Normalize(2)(embed_g)
    local batch_g = nn.View(-1, opt.N*opt.k, opt.n_kernels)(norm_g)
    
    -- in: (B x kB x 64) , (B x N*k x 64)
    --   -> MM: -> B x N*k x kB
    --   -> Transpose: B x kB x N*k
    --   -> View -> (B * kB) x N*k
    --   -> SoftMax -> (B * kB) x N*k
    --   -> View -> B x kB x N*k
    --   -> Transpose -> B x N*k x kB
    --   -> IndexAdd -> B x N x kB
    --   -> Transpose -> B x kB x N
    --   -> View -> B*kB x N
    local cos_dist = nn.MM(false, true)({batch_g, batch_f})
    local unbatch1 = nn.View(-1, opt.N*opt.k)(nn.Transpose({2,3})(cos_dist))
    local attn_scores = nn.SoftMax()(unbatch1)
    local rebatch = nn.Transpose({2,3})(nn.View(-1, opt.kB, opt.N*opt.k)(attn_scores))
    local class_probs = nn.IndexAdd(1, opt.N)({rebatch, inputs[3]})
    local unbatch2 = nn.View(-1, opt.N)(nn.Transpose({2,3})(class_probs))
    local log_probs = nn.Log()(unbatch2)
    local outputs = {log_probs}
    local crit = nn.ClassNLLCriterion()

    return nn.gModule(inputs, outputs), crit
end

function make_baseline(opt)
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
    local linear = nn.Linear(opt.n_kernels, n_classes)(embed_f)
    local outputs = {linear}
    local crit = nn.CrossEntropyCriterion()

    return nn.gModule(inputs, outputs), crit
end

function make_baseline_nearest_neighbor(opt, embed)
    local inputs = {}
    table.insert(inputs, nn.Identity()())
    
end

function make_cnn(opt)
    local input = nn.Identity()()
    local layers = {}
    for i=1,opt.n_modules do
        if i == 1 then
            local conv_module = make_cnn_module(opt, 1)
            conv_module.name = 'cnn' .. i
            table.insert(layers, conv_module(input))
        else
            local conv_module = make_cnn_module(opt, opt.n_kernels)
            conv_module.name = 'cnn' .. i
            table.insert(layers, conv_module(layers[i-1]))
        end
    end
    local output = layers[opt.n_modules]
    return nn.gModule({input}, {output})
end

function make_cnn_module(opt, n_input_feats)
    local conv_w = opt.conv_width
    local conv_h = opt.conv_height
    local pad_w = opt.conv_pad --math.floor((conv_w - 1) / 2)
    local pad_h = opt.conv_pad --math.floor((conv_h - 1) / 2)
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
        output = pool_layer(nn.ReLU()(norm_layer))
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
