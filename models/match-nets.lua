function make_matching_net(opt)
    -- input is support set and labels, test datum
    local input = {}
    table.insert(input, nn.Identity()()) -- hat(x)
    table.insert(input, nn.Identity()()) -- x_i; TODO can make this input a table?
    table.insert(input, nn.Identity()()) -- y_i
    
    local output = {}
    local f = make_cnn(opt)
    f.name = 'embed_f'
    local embed_f = f(input[1])

    local g = make_cnn(opt)
    g.name = 'embed_g'
    local embed_g = nn.Sequential()
    embed_g:add(nn.SplitTable(1)(input[2]))     -- split up big tensor into table of tensors
    embed_g:add(nn.MapTable(g))                 -- embed each g then
    embed_g:add(nn.MapTable(nn.Normalize(2)))   -- normalize for cosine similarity
    embed_g:add(nn.JoinTable())                 -- join into one giant matrix
    
    -- cosine distance between g's and f
    table.insert(output, nn.MM(embed_f, embed_g)) -- NOTE: not normalizing f currently

    return nn.gModule({input}, {output})

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
        local norm_layer = cudnn.BatchNormalization(
            opt.num_kernels)(conv_layer)
        local pool_layer = cudnn.SpatialMaxPooling(
            opt.pool_width, opt.pool_height)(nn.ReLU()(norm_layer))
        output = (pool_layer)
    else
        local conv_layer = nn.SpatialConvolution(input_size, opt.n_kernels,
            opt.conv_width, opt.conv_height)(nn.View()(input))
        local norm_layer = nn.BatchNormalization(
            opt.num_kernels)(conv_layer)
        local pool_layer = nn.SpatialMaxPooling(
            opt.pool_width, opt.pool_height)(nn.ReLU()(norm_layer))
        output = (pool_layer)
    end
    return nn.gModule({input},{output})
end
