package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'

local Data = torch.class("Data")

function Data:__init(opt, datasets)
    self.gpuid = opt.gpuid
    self.xs = datasets[1]
    self.ys = datasets[2]:long()
    self.embedding_fn = opt.embedding_fn
    if opt.embedding_fn == 'cnn' then -- would be clearer just to ResizeAs
        self.n_channels = opt.n_channels
        self.im_dim = opt.im_dim
        assert(self.xs:size()[3] == self.n_channels, 'only 1d layer supported')
        assert(self.xs:size()[4] == self.im_dim, 'only 1d layer supported')
        assert(self.xs:size()[5] == self.im_dim, 'only 1d layer supported')
    else
        self.seq_len = opt.seq_len
    end
    self.k = opt.k 
    self.N = opt.N 
    self.kB = opt.kB 
    self.batch_size = opt.batch_size
    self.n_episodes = self.xs:size(1)
    self.n_batches = self.n_episodes / self.batch_size
    self.perm = torch.range(1, self.n_batches):long()
end

function Data.__index(self,idx)
    local input, targ
    if type(idx) == "string" then
        return Data[idx]
    else
        if idx > self.n_batches then
            return "Error: invalid batch size"
        end
        local B = self.batch_size
        local n_channels = self.n_channels
        local im_dim = self.im_dim
        local set_size = self.N * self.k
        local bat_size = self.kB
        local shuffle

        local p_idx = self.perm[idx] -- shuffle batch order
        local meta_xs = self.xs:narrow(1,(p_idx-1)*B+1,B)
        local meta_ys = self.ys:narrow(1,(p_idx-1)*B+1,B)
        local set_xs, bat_xs
        if self.embedding_fn == 'cnn' then
            set_xs = torch.zeros(B*set_size, n_channels, im_dim, im_dim)
            bat_xs = torch.zeros(B*bat_size, n_channels, im_dim, im_dim)
        else
            set_xs = torch.zeros(B*set_size, opt.seq_len)
            bat_xs = torch.zeros(B*bat_size, opt.seq_len)
        end
        local set_ys = torch.zeros(B,set_size):long()
        local bat_ys = torch.zeros(B*bat_size):long() -- one-dim tensor for easy evaluation

        for i=1,B do -- shuffle within episode
            --shuffle = torch.randperm(set_size):long()
            shuffle = torch.range(1,set_size):long() 
            set_xs:narrow(1,(i-1)*set_size+1,set_size):index(meta_xs[i]:narrow(1,1,set_size),1,shuffle)
            set_ys[i]:index(meta_ys[i]:narrow(1,1,set_size),1,shuffle)

            shuffle = torch.randperm(bat_size):long()
            --shuffle = torch.range(1,bat_size):long() 
            bat_xs:narrow(1,(i-1)*bat_size+1,bat_size):index(meta_xs[i]:narrow(1,set_size+1,bat_size),1,shuffle)
            bat_ys:narrow(1,(i-1)*bat_size+1,bat_size):index(meta_ys[i]:narrow(1,set_size+1,bat_size),1,shuffle)
        end

        if self.gpuid > 0 then
            set_xs = set_xs:cuda()
            bat_xs = bat_xs:cuda()
            set_ys = set_ys:cuda()
            bat_ys = bat_ys:cuda()
        end
        input = {bat_xs, set_xs, set_ys}
        targ = bat_ys
        return {input, targ}
    end
end

function Data.shuffle()
    self.perm = torch.randperm(self.n_batches):long()
end
