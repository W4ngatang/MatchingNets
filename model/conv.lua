
function make_matching_net(opt)

end

function make_cnn(opt)
    local model = nn.Sequential()
    for i=1,opt.n_modules do
        -- probably want to name the layer
        model:add(make_cnn_module(opt))
    end
end

function make_cnn_module(opt)
    nn.SpatialConvolution(nInputLayers, nOutputLayers , opt.conv_width, opt.conv_height)
    nn.BatchNormalization()
    nn.ReLU()
    nn.SpatialMaxPooling(opt.pool_width, opt.pool_height)
end
