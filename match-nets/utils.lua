package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'

--
-- Various models used throughout experiments
--

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
