# MatchingNets
Re-implementation of ["Matching Nets for One-Shot Learning"](https://arxiv.org/abs/1606.04080). Project is currently in progress; it runs but the test accuracy is a bit below the numbers reported in the original work.

The setup and training is all done in main.lua(). The code implementing the matching network is in match-nets/. The debugger.lua/ folder contains a debugger from [here](https://github.com/slembcke/debugger.lua) that is useful because Lua does not seem to have a commandline debugger built in. The data/ folder should contain the Omniglot or other image data, and the omniglot_preprocess.py script resizes the images from 105x105 to 28x28, following the original work.
