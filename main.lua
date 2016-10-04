require 'torch'
require 'nn'
require 'hdf5'

require 'XXXX.models'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a matching network')
cmd:text('Options')
cmd:option('-seed', 42, 'random seed')

function train()

end

function eval()

end

function main()
    opt = cmd:parse(arg)
    torch.manualSeed(opt.seed)

    if opt.gpuid >= 0 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        require 'cutorch'
        require 'cunn'
        if opt.cudnn == 1 then
            assert(opt.gpuid >= 0, 'GPU must be used if using cudnn')
            print('using cudnn...')
            require 'cudnn'
        end
        cutorch.setDevice(opt.gpuid+1)
        cutorch.manualSeed(opt.seed)
   end

   -- load data
   
   -- build model
end
