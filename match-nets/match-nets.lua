--
-- Various models used throughout experiments
--


function make_matching_net(opt)
    -- input is support set and labels, test datum
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- hat(x)
    table.insert(inputs, nn.Identity()()) -- x_i; TODO can this be a table?
    --table.insert(inputs, nn.Identity()()) -- y_i; note: not used currently
                                         -- TODO weight indices by score
    
    local outputs = {}
    local f = make_cnn(opt)
    f.name = 'embed_f'
    local embed_f = f(inputs[1]) -- TODO normalize f

    local g = make_cnn(opt)
    g.name = 'embed_g'
    local embed_g = nn.JoinTable(1)(
        nn.MapTable(nn.Normalize(2))(
        nn.MapTable(g)(nn.SplitTable(1)(inputs[2]))))
    
    -- cosine distance between g's and f
    table.insert(outputs, nn.MM()({embed_f, embed_g})) 
    return nn.gModule(inputs, outputs)

end

function make_cnn(opt)
    local input = nn.Identity()()
    local layers = {}
    for i=1,opt.n_modules do
        -- probably want to name the layer
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
    return nn.gModule({input}, {output}), layers
end

function make_cnn_module(opt, input_size)
    local output
    local input = nn.Identity()() -- double () denotes nngraph module
    if opt.cudnn == 1 then
        -- 1 is the number of input channels since b&w
        local conv_layer = cudnn.SpatialConvolution(input_size , opt.n_kernels, 
            opt.conv_width, opt.conv_height)(nn.View()(input))
        local norm_layer = cudnn.BatchNormalization(opt.n_kernels)(conv_layer)
        local pool_layer = cudnn.SpatialMaxPooling(
            opt.pool_width, opt.pool_height)(nn.ReLU()(norm_layer))
        output = (pool_layer)
    else
        local conv_layer = nn.SpatialConvolution(input_size, opt.n_kernels,
            opt.conv_width, opt.conv_height)(nn.View()(input))
        local norm_layer = nn.BatchNormalization(opt.n_kernels)(conv_layer)
        local pool_layer = nn.SpatialMaxPooling(
            opt.pool_width, opt.pool_height)(nn.ReLU()(norm_layer))
        output = (pool_layer)
    end
    return nn.gModule({input},{output})
end
