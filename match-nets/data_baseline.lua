package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'

--[[ 
--
-- Data loader class for training baseline model
-- of class prediction given a single image
--
--]]

local DataBaseline = torch.class("DataBaseline")

function DataBaseline:__init(opt, datasets)
    self.gpuid = opt.gpuid
    self.xs = datasets[1]:double()
    self.ys = datasets[2]:long()
    self.k = opt.k 
    self.N = opt.N 
    self.kB = opt.kB 
    self.batch_size = opt.batch_size
    self.n_episodes = self.xs:size(1)
    self.n_batches = self.n_episodes / self.batch_size
    self.batch_inputs = torch.DoubleTensor()
    self.batch_outputs = torch.LongTensor()
    if opt.gpuid > 0 then
        self.batch_inputs = self.batch_inputs:cuda()
        self.batch_outputs = self.batch_outputs:cuda()
    end
    self.perm = torch.range(1, self.n_batches)
end

function DataBaseline.__index(self,idx)
    local input, targ
    if type(idx) == "string" then
        return data_baseline[idx]
    else
        if idx > self.n_batches then
            return "Error: invalid batch size"
        end
        local B = self.batch_size
        local shuffle = torch.randperm(B):long()
        local p_idx = self.perm[idx] -- shuffle batch order
        local inputs = self.xs:narrow(1,(p_idx-1)*B+1,B)
        local outputs = self.ys:narrow(1,(p_idx-1)*B+1,B)
        if self.gpuid > 0 then
            inputs = inputs:cuda()
            outputs = outputs:cuda()
        end
        self.batch_inputs:resizeAs(inputs):index(inputs, 1, shuffle)
        self.batch_outputs:resizeAs(outputs):index(outputs, 1, shuffle)

        return {self.batch_inputs, self.batch_outputs}
    end
end
