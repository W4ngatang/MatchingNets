--[[ Module to perform indexAdd operation ALONG ROWS
]]

package.path = package.path .. ";" .. os.getenv("HOME") .. 
    "/MatchingNets/match-nets/?.lua" .. ";" .. os.getenv("HOME")
    .. "/MatchingNets/debugger.lua/?.lua"
local dbg = require 'debugger'

local IndexAdd, parent = torch.class('nn.IndexAdd', 'nn.Module')

-- N: length of output along dimension dim
function IndexAdd:__init(dim, N)
    parent.__init(self)
    self.dim = dim
    self.gradInput = {self.gradInput, self.gradInput.new()}
    self.N = N
end

-- in: 
--      t: B x p x q
--      inds: B x p
-- out: B x N x q
function IndexAdd:updateOutput(input)
    assert(#input == 2, 'input must be a tensor and indices')
    local t, inds = table.unpack(input)
    assert(t:nDimension() == 2 or t:nDimension() == 3, 'input tensors must be 2D or 3D')

    if t:nDimension() == 2 then
        assert(inds:nDimension() == 1, 'indices must be 1D')
        assert(inds:size(1) == t:size(1), 'tensor sizes do not match')
        self.output:resize(self.N, t:size(2)):zero()
        self.output:indexAdd(self.dim, inds, t)
    else
        assert(inds:nDimension() == 2, 'indices must be 2D')
        assert(inds:size(1) == t:size(1), 'inputs must contain same number of minibatches')
        assert(inds:size(2) == t:size(2), 'tensor sizes do not match')
        self.output:resize(t:size(1), self.N, t:size(2)):zero()
        for i = 1, t:size(1) do
            self.output:narrow(1,i,1):indexAdd(self.dim, inds[i], t[i])
        end
    end
    return self.output
end

function IndexAdd:updateGradInput(input, gradOutput)
    assert(#input == 2, 'input must be a tensor and indices')
    local t, inds = table.unpack(input)
    dbg()

    assert(gradOutput:nDimension() == 2 or gradOutput:nDimension() == 3, 'arguments must be a 2D or 3D tensor')

    self.gradInput[2]:resizeAs(inds):zero()
    local gradInput = self.gradInput[1]
    gradInput:resizeAs(t):zero()
    if gradOutput:nDimension() == 2 then
        assert(t:nDimension() == 2, 'input tensor must be 2D')
        assert(inds:nDimension() == 1, 'index tensor must be 1D')
        for i = 1, inds:nElement() do -- do reverse indexAdd
            gradInput:narrow(1,i,1):add(gradOutput:narrow(1,inds[i],1))
        end
    else
        assert(t:nDimension() == 3, 'input tensor must be 3D')
        assert(inds:nDimension() == 2, 'index tensor must be 2D')
        for b = 1, t:size(1) do             -- for each batch
            for i = 1, inds:nElement() do   -- do reverse indexAdd
                gradInput:sub(b,b,i,i):add(gradOutput:sub(b,b,inds[b][i],inds[b][i]))
            end
        end
    end
    return self.gradInput
end

function IndexAdd:clearState()
    self.gradInput[1]:set()
    self.gradInput[2]:set()
    self.output:set()
    return self
end
