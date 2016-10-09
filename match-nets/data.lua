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
end

function data.__index(self,idx)
    local input, targ
    if type(idx) == "string" then
        return data[idx]
    else
        -- TODO shuffle within episode
        set_xs = self.xs[idx]:narrow(1,1,self.k) 
        bat_xs = self.xs[idx]:narrow(1,self.k+1,self.kB)
        set_ys = self.ys[idx]:narrow(1,1,self.k)
        bat_ys = self.ys[idx]:narrow(1,self.k+1,self.kB)
        if self.gpuid >= 0 then
            set_xs = set_xs:cuda()
            bat_xs = bat_xs:cuda()
            set_ys = set_ys:cuda()
            bat_ys = bat_ys:cuda()
        end
        input = {set_xs, bat_xs}--, set_ys}
        targ = bat_ys
    end
    return {input, target}
end
