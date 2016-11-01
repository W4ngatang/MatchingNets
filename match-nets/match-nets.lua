--
-- Various models used throughout experiments
--

function make_matching_net(opt)
    -- input is support set and labels, test datum
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- hat(x); shape kb x 1 x im x im
    table.insert(inputs, nn.Identity()()) -- support set: shape k x 1 x im x im
    table.insert(inputs, nn.Identity()()) -- y_i
    local outputs = {}

    -- in: N*kB x im x im
    --   -> Unsqueeze -> N*kB x 1 x im x im
    --   -> f -> N*kB x 64 x 1 x 1
    --   -> Squeeze -> N*kB x 64
    --   -> normalize -> N*k x 64
    -- out: N*kB x 64
    local f = make_cnn(opt)
    f.name = 'embed_f'
    local embed_f = nn.Squeeze()(f(nn.Unsqueeze(2)(inputs[1])))
    local norm_f = nn.Normalize(2)(embed_f)

    -- in: N*k x im x im
    --   -> Unsqueeze -> N*k x 1 x im x im
    --   -> g -> N*k x 64 x 1 x 1
    --   -> Squeeze -> N*k x 64
    --   -> normalize -> N*k x 64
    -- out: N*k x 64
    local g = nil
    if opt.share_embed == 1 then
        print('\tTying embedding function parameters...')
        g = f
    else
        g = make_cnn(opt)
        g.name = 'embed_g'
    end
    local embed_g = nn.Squeeze()(g(nn.Unsqueeze(2)(inputs[2])))
    local norm_g = nn.Normalize(2)(embed_g)
    
    local match_scores = nn.MM2(false, true)({norm_f, norm_g}) --(N*k) x (N*kb)
    local class_scores = nn.IndexAdd(1, opt.N, opt.kB*opt.N)({match_scores, inputs[3]}) -- N x (N*kb)

    local crit = nil
    if opt.match_fn == 'softmax' then
        -- maybe optional softmax? should this be taken over ALL (x_i, y_i)?
        local probs = nn.LogSoftMax()(nn.Transpose({1,2})(class_scores))
        table.insert(outputs, probs)
        crit = nn.ClassNLLCriterion()
    elseif opt.match_fn == 'cosine' then
        table.insert(outputs, class_scores)
        crit = nn.CosineEmbeddingCriterion()
    else
        print("Matching function " .. opt.match_fn .. " not supported!")
        return
    end

    return nn.gModule(inputs, outputs), crit

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
    local pad_w = math.floor((conv_w - 1) / 2)
    local pad_h = math.floor((conv_h - 1) / 2)
    local pool_w = opt.pool_width
    local pool_h = opt.pool_height

    local output
    local input = nn.Identity()() -- double () denotes nngraph module
    if opt.cudnn == 1 then
        -- 1 is the number of input channels since b&w
        -- conv height and width change every layer
        local conv_layer = cudnn.SpatialConvolution(n_input_feats, opt.n_kernels, 
            conv_w, conv_h, 1, 1, pad_w, pad_h)(input)
        local norm_layer = cudnn.SpatialBatchNormalization(opt.n_kernels)(conv_layer)
        local pool_layer = cudnn.SpatialMaxPooling(
            pool_w, pool_h, pool_w, pool_h)(nn.ReLU()(norm_layer))
        output = (pool_layer)
    else
        local conv_layer = nn.SpatialConvolution(n_input_feats, opt.n_kernels,
            conv_w, conv_h, 1, 1, pad_w, pad_h)(input)
        local norm_layer = nn.SpatialBatchNormalization(opt.n_kernels)(conv_layer)
        local pool_layer = nn.SpatialMaxPooling(
            pool_w, pool_h, pool_w, pool_h)(nn.ReLU()(norm_layer))
        output = (pool_layer)
    end
    return nn.gModule({input},{output})
end
