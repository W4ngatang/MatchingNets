--[[ Module to perform indexAdd operation ALONG ROWS
]]

package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'

local IndexAdd, parent = torch.class('nn.IndexAdd', 'nn.Module')

-- N: length of output along dimension dim
function IndexAdd:__init(dim, N)--, n_bat)
    parent.__init(self)
    self.dim = dim
    self.gradInput = {self.gradInput, self.gradInput.new()}
    self.N = N
    --self.n_bat = n_bat
end

-- in: 
--      t: B x p x q
--      inds: B x p
-- out: B x N x q
function IndexAdd:updateOutput(input)
    local t = input[1]
    local inds = input[2]
    --self.output:resize(self.N, self.n_bat):zero()
    self.output:resize(self.N, t:size(2)):zero()
    self.output:indexAdd(self.dim, inds, t)
    return self.output
end

function IndexAdd:updateGradInput(input, gradOutput)
    local t = input[1]
    local inds = input[2]

    self.gradInput[2]:resize(inds:size()):zero()
    local gradInput = self.gradInput[1]
    gradInput:resizeAs(t):zero()
    for i = 1, inds:nElement() do -- do reverse indexAdd
        gradInput:narrow(1,i,1):add(gradOutput:narrow(1,inds[i],1))
    end
    return self.gradInput
end

function IndexAdd:clearState()
    self.gradInput[1]:set()
    self.gradInput[2]:set()
    self.output:set()
    return self
end
