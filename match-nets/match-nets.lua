--
-- Various models used throughout experiments
--

function make_matching_net(opt)
    -- input is support set and labels, test datum
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- hat(x): B*N*kb x im x im
    table.insert(inputs, nn.Identity()()) -- x_i: B*N*k x im x im
    table.insert(inputs, nn.Identity()()) -- y_i: B x N*k

    -- in: B*N*kB x im x im
    --   -> unsqueeze -> B*N*kB x 1 x im x im
    --   -> f -> B*N*kB x 64 x 1 x 1
    --   -> squeeze -> B*N*kB x 64
    --   -> normalize -> B*N*kB x 64
    --   -> reshape -> B x N*kB x 64
    -- out: B x N*kB x 64
    local f = make_cnn(opt)
    f.name = 'embed_f'
    local embed_f = nn.Squeeze()(f(nn.Unsqueeze(2)(inputs[1])))
    local norm_f = nn.Normalize(2)(embed_f)
    local batch_f = nn.View(-1, opt.N*opt.kB, opt.n_kernels)(norm_f)

    -- in: B*N*k x im x im
    --   -> unsqueeze -> B*N*k x 1 x im x im
    --   -> g -> B*N*k x 64 x 1 x 1
    --   -> squeeze -> B*N*k x 64
    --   -> normalize -> B*N*k x 64
    --   -> view -> B x N*k x 64
    -- out: B x N*k x 64
    local g
    if opt.share_embed == 1 then
        g = f
    else
        g = make_cnn(opt)
        g.name = 'embed_g'
    end
    local embed_g = nn.Squeeze()(g(nn.Unsqueeze(2)(inputs[2])))
    local norm_g = nn.Normalize(2)(embed_g)
    local batch_g = nn.View(-1, opt.N*opt.k, opt.n_kernels)(norm_g)
    
    -- in: (B x N*kB x 64) , (B x N*k x 64)
    --   -> MM: -> B x N*k x N*kB
    --   -> View -> (B * N*kB) x N*k
    --   -> SoftMax -> (B * N*kB) x N*k
    --   -> View -> B x N*k x N*kB
    --   -> IndexAdd -> B x N x N*kb
    --   -> Transpose -> B x N*kB x N
    --   -> View -> B*N*kB x N
    local cos_dist = nn.MM2(false, true)({batch_g, batch_f})
    local unbatch1 = nn.View(-1, opt.N*opt.k)(cos_dist)
    local attn_scores = nn.SoftMax()(unbatch1)
    local rebatch = nn.View(-1, opt.N*opt.k, opt.N*opt.kB)(attn_scores)
    local class_scores = nn.IndexAdd(1, opt.N)({rebatch, inputs[3]})
    local unbatch2 = nn.View(-1, opt.N)(nn.Transpose({2,3})(class_scores))
    local log_probs = nn.Log2()(unbatch2)
    local outputs = {log_probs}
    local crit = nn.ClassNLLCriterion()

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
        local norm_layer = cudnn.SpatialBatchNormalization(opt.n_kernels, nil, nil)(conv_layer)
        local pool_layer = cudnn.SpatialMaxPooling(
            pool_w, pool_h, pool_w, pool_h)(nn.ReLU()(norm_layer))
        output = pool_layer
    else
        local conv_layer = nn.SpatialConvolution(n_input_feats, opt.n_kernels,
            conv_w, conv_h, 1, 1, pad_w, pad_h)(input)
        local norm_layer = nn.SpatialBatchNormalization(opt.n_kernels)(conv_layer)
        local pool_layer = nn.SpatialMaxPooling(
            pool_w, pool_h, pool_w, pool_h)(nn.ReLU()(norm_layer))
        output = pool_layer
    end
    return nn.gModule({input},{output})
end
