
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

    local inds = torch.range(1, self.N):reshape(self.N, 1):long()
    local set_ys = inds:repeatTensor(1,self.k):view(-1):long()
    local bat_ys = inds:repeatTensor(1,self.kB):view(-1):long()
    --local set_zs = torch.zeros(self.k*self.N, self.N)
    --local bat_zs = torch.zeros(self.kB*self.N, self.N)
    self.set_ys = set_ys --set_zs:scatter(2, set_ys, 1)
    self.bat_ys = bat_ys --bat_zs:scatter(3, bat_ys, 1)
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
                - batching, something like :narrow(1,idx,batch_size):narrow(2,1, N*k)
        --]]

        local shuffle = torch.randperm(self.N * self.k):type('torch.LongTensor')
        local set_xs = self.xs[idx]:narrow(1,1,self.N*self.k):index(1, shuffle)
        local set_ys = self.set_ys:index(1, shuffle)

        local shuffle = torch.randperm(self.N * self.kB):type('torch.LongTensor')
        local bat_xs = self.xs[idx]:narrow(1,self.N*self.k+1,self.N*self.kB):index(1, shuffle)
        local bat_ys = self.bat_ys:index(1, shuffle)
        --set_ys = torch.squeeze(self.ys[idx]:narrow(1,1,self.N*self.k))
        --bat_ys = torch.squeeze(self.ys[idx]:narrow(1,self.N*self.k+1,self.N*self.kB))

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
