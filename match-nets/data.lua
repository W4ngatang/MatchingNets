
package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'

local data = torch.class("data")

function data:__init(opt, datasets)
    self.gpuid = opt.gpuid
    self.xs = datasets[1]
    self.ys = datasets[2]
    self.k = opt.k 
    self.N = opt.N 
    self.kB = opt.kB 
    self.batch_size = opt.batch_size
    self.n_episodes = self.xs:size(1)
    self.n_batches = self.n_episodes / self.batch_size
    self.perm = torch.randperm(self.n_batches):type('torch.LongTensor')

    local inds = torch.range(1, self.N):reshape(self.N, 1):long()
    local set_ys = inds:repeatTensor(1,self.k):view(-1):long()
    local bat_ys = inds:repeatTensor(1,self.kB):view(-1):long()
    self.set_ys = set_ys 
    self.bat_ys = bat_ys
end

function data.__index(self,idx)
    local input, targ
    if type(idx) == "string" then
        return data[idx]
    else
        if idx > self.n_batches then
            return "Error"
        end
        --[[ TODO
        - batching, something like :narrow(1,idx,batch_size):narrow(2,1, N*k
        --]]

        p_idx = self.perm[idx]
        local shuffle = torch.randperm(self.N * self.k):type('torch.LongTensor')
        local set_xs = self.xs[p_idx]:narrow(1,1,self.N*self.k):index(1, shuffle)
        local set_ys = self.set_ys:index(1, shuffle)

        shuffle = torch.randperm(self.N * self.kB):type('torch.LongTensor')
        local bat_xs = self.xs[p_idx]:narrow(1,self.N*self.k+1,self.N*self.kB):index(1, shuffle)
        local bat_ys = self.bat_ys:index(1, shuffle)

        if self.gpuid > 0 then
            set_xs = set_xs:cuda()
            bat_xs = bat_xs:cuda()
            set_ys = set_ys:cuda()
            bat_ys = bat_ys:cuda()
        end
        input = {set_xs, bat_xs, set_ys}
        targ = bat_ys
    end
    return {input, targ}
end

-- actually don't need to call this because on shuffling perm is set to random
function data.shuffle()
    self.perm = torch.randperm(self.n_batches):type('torch.LongTensor')
end
