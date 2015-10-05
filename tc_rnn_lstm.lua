local whohow = [[
---------------------------------------------------------------------------------

             
              |       |     
              |--\  /-|-.-.             .
              |__/\/  | \/_            . :.
               ___/ _   _  _  __   ___/ . .
              / ._|| | | || \/  \ /   \ . : .: 
              \_  \| |_| || | | || [ ] | .. 
              \___/|____/||_|_|_| \___/ ..
                                  / . ..  .:
                                   .: .  .:  
                


  Example LSTM experiements

  Copyright 2015, ByteSumo Limited (Andrew J Morgan)

  Author:  Andrew J Morgan
  Version: 1.0.0
  License: GPLv3

    This file is part of TrendCalculus.

    TrendCalculus is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    TrendCalculus is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with TrendCalculus.  If not, see <http://www.gnu.org/licenses/>.

Instructions for Use:

Copy this script and modify the working parameters for:
  - your csv filename
  - the number of input, hidden, and output nodes
  - the field mappings, between your csv and internal variables
  - to include any transformations needed to your raw data, to prepare it for the RNN
  - to set out the training OFFSET treshold. This will train in one pass up to threshold.
  - run the script using torch, not plain luajit. www.torch.ch
  
Note:

This program will spool your CSV records through an LSTM, set up in training mode until
 it hits a threshold you set using the OFFSET variable. 

After that OFFSET, it will stop training, and then run the LSTM in Test mode, printing the
network's predictions alongside the training target "output" values for inspection.

---------------------------------------------------------------------------------------

]]

-- print(whohow)
-- require('mobdebug').start()

--[[ require libraries ]]--
        -- require 'nn'
        -- require 'dp'
        --require 'nngraph'


local csv = require("csv")
require 'rnn'
require 'cltorch'
require 'clnn'
require 'pprint'

--[[ configure csv reading]]--

local filename = "dp_10_train_DJI_min.csv"               
local csvparams ={}

csvparams.header = true
csvparams.separator = ","

 -- timeseries data in csv note:
 -- use this params list to map columns/headers in your csv file to our internal variable names. 
 -- Note desired "output" is to be mapped to the field in the file called "trend" but we need to also
 -- do some preprocessing to convert it from a char binomial output to a number output on two output nodes, i.e.: [1,0] or [0,1]
 
csvparams.columns = {  -- we list possible header aliases, and map them to our stable var
                    tseries  = { names = {"TCFG", "symbol", "instrument", "id"}},  
                    percent_change  = { names = {"unit_per_change", "percent_change", "relative_change"}},
                    char_output  = { names = {"trend"}},
                    dummy  = { names = {"dummy"}},
                    t1  = { names = {"T1"}},
                    t2  = { names = {"T2"}},
                    t3  = { names = {"T3"}},
                    t4  = { names = {"T4"}},
                    t5  = { names = {"T5"}},
                    t6  = { names = {"T6"}},
                    t7  = { names = {"T7"}},
                    t8  = { names = {"T8"}},
                    t9  = { names = {"T9"}},
                    t10  = { names = {"T10"}},
                    t11  = { names = {"T11"}},
                    t12  = { names = {"T12"}},
                    t13  = { names = {"T13"}},
                    t14  = { names = {"T14"}},
                    t15  = { names = {"T15"}},
                    t16  = { names = {"T16"}},
                    t17  = { names = {"T17"}},
                    t18  = { names = {"T18"}},
                    t19  = { names = {"T19"}},
                    t20  = { names = {"T20"}},
                    t21  = { names = {"T21"}},
                    t22  = { names = {"T22"}},
                    t23  = { names = {"T23"}},
                    t24  = { names = {"T24"}},
                    t25  = { names = {"T25"}},
                    t26  = { names = {"T26"}},
                    t27  = { names = {"T27"}}
                 }                 
                 


--[[ model definition params ]]--

local threshold = 100000  -- train up to this offset in the file, then predict.
local inputSize = 28      -- the number features in the csv file to use as inputs 
local hiddenSize = 50     -- the number of hidden nodes
local outputSize = 2      -- the number of outputs representing the prediction targat. 
local dropoutProb = 0.2   -- a dropout probability, might help with generalisation
local rho = 50            -- the timestep history we'll recurse back in time when we apply gradient learning
local batchSize = 200     -- the size of the episode/batch of sequenced timeseries we'll feed the rnn
local lr = 0.001          -- the learning rate to apply to the weights

--[[ build up model definition ]]--

-- create a trend guessing model, "tmodel"
tmodel = nn.Sequential()   -- wrapping it all in Sequential brings forward / backward methods to all layers in one go
tmodel:add(nn.Sequencer(nn.Identity())) -- untransfomed input data
tmodel:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize, rho)))  -- will create a complex network of lstm cells that learns back rho timesteps
tmodel:add(nn.Sequencer(nn.Dropout(dropoutProb)))                  -- I'm sticking in a place to do dropout, not strickly needed I don't think
tmodel:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize, rho))) -- creating a second layer of lstm cells, coz I can 
tmodel:add(nn.Sequencer(nn.Linear(hiddenSize, outputSize)))        -- reduce the output back down to the output class nodes
tmodel:add(nn.Sequencer(nn.LogSoftMax()))                          -- apply the log soft max, as we're guessing classes. 


--criterion = nn.SequencerCriterion(nn.MSECriterion())      -- if you want a regression, use mse
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())   -- if you want a classifier, use this. 
                                                            -- note lua arrays indexed at 1, so binary class flags are like [1,2], [2,1] not [0,1],[1,0]
--[[ set out learning functions ]]--

-- code borrowed from an example by fabio. thanks!
function gradientUpgrade(model, x, y, criterion, learningRate, i)
	local prediction = model:forward(x)
	local err = criterion:forward(prediction, y)
   if i % 10 == 0 then
      print('error for iteration ' .. i  .. ' is ' .. err/rho)
   end
	local gradOutputs = criterion:backward(prediction, y)
  if err > 0 then
	model:backward(x, gradOutputs)
	model:updateParameters(learningRate)
  model:zeroGradParameters()
  end
end

--[[ FEED THE NETWORK WITH VALUES ]]--

-- initialise some of the file counters
local offset = 0
local batchcounter = 0
local episode = 0

-- Open the file iterator. -- this is a streamed csv parser, it doesn't read the file into ram first, but in chunks (like sax parsing on xml....)
local f = csv.open(filename, csvparams)

-- create structured lua tables from csv. for the moment they are global till I figure stuff out

feed_input = {}     -- create a table to hold our input data we'll use to predict, I want a table of tensors
feed_output = {}    -- create a table to hold our target outputs to train against
feed_offsets = {}   -- create a table to hold our batch offset references -- not sure I need this

for line in f:lines() do              -- the file iterator, iterates by lines in my csv raw data file
    offset = offset + 1               -- this is a global file offset, counts the rownumber of the raw file
    
                -- inputs to learn from --
                batchcounter = batchcounter +1    -- this tracks the offset inside an episode (aka chunk of time, the max length of which is rho)
                -- notes:
                -- We load up our lua table, row by row indexed 1 .. batchSize, and for each row, we insert a tensor of input records. 
                feed_input[batchcounter] = torch.Tensor(  
                                          {  -- insert inputs into standard training dataset table, indexed by this offset
                                            tonumber(line.percent_change)             
                                          , tonumber(line.t1)
                                          , tonumber(line.t2)
                                          , tonumber(line.t3)
                                          , tonumber(line.t4)
                                          , tonumber(line.t5)
                                          , tonumber(line.t6)
                                          , tonumber(line.t7)
                                          , tonumber(line.t8)
                                          , tonumber(line.t9)
                                          , tonumber(line.t10)
                                          , tonumber(line.t11)
                                          , tonumber(line.t12)
                                          , tonumber(line.t13)
                                          , tonumber(line.t14)
                                          , tonumber(line.t15)
                                          , tonumber(line.t16)
                                          , tonumber(line.t17)
                                          , tonumber(line.t18)
                                          , tonumber(line.t19)
                                          , tonumber(line.t20)
                                          , tonumber(line.t21)
                                          , tonumber(line.t22)
                                          , tonumber(line.t23)
                                          , tonumber(line.t24)
                                          , tonumber(line.t25)
                                          , tonumber(line.t26)
                                          , tonumber(line.t27)
                                         }
                                        )
                 
                --[[ targets to learn ]]--
                
                -- construct flags that map to our training labels. (Remember classes indexed from 1 .. n), so our
                -- char class of [U,D] becomes {{2,1},{1,2}} 
                
                if line.char_output == "U" then                           
                      feed_output[batchcounter] = torch.Tensor({2, 1})    
                else                                                     
                      feed_output[batchcounter] = torch.Tensor({1, 2})    
                end   
                
        --[[ TRAIN THE RNN ]]-- (note we are still inside the file iterator here)

        if offset < threshold and offset % batchSize == 0 then    -- bactSize defines the episode size
          episode = episode + 1                                   -- create a counter that tracks the total number of episodes we've created
          
          -- now send this batch episode to rnn backprop through time learning
         
          gradientUpgrade(tmodel, feed_input, feed_output, criterion, lr, episode)
          
          -- now clear out the tables, reset them     
          feed_input = nil; feed_input = {}
          feed_output = nil; feed_output = {}
          feed_offsets = nil; feed_offsets = {}
          
          -- reset the rowID of the batch episode back to zero
          batchcounter = 0
          
        end -- end of the batchSize full event trigger 
        
        --[[ Validation ]]--
        
        if offset > threshold then        -- we have now rolled through the timeseries learning, but can we guess the ending?   
                                          -- note we are still inside the file iterator. 
            -- TEST OF PREDICTION --            
            -- grab the current row of inputs, and generate prediction
            
            local realtime_input = {}
            realtime_input[1] = torch.Tensor(  
                                          {  -- insert inputs into standard training dataset table, indexed by this offset
                                            tonumber(line.percent_change)             
                                          , tonumber(line.t1)
                                          , tonumber(line.t2)
                                          , tonumber(line.t3)
                                          , tonumber(line.t4)
                                          , tonumber(line.t5)
                                          , tonumber(line.t6)
                                          , tonumber(line.t7)
                                          , tonumber(line.t8)
                                          , tonumber(line.t9)
                                          , tonumber(line.t10)
                                          , tonumber(line.t11)
                                          , tonumber(line.t12)
                                          , tonumber(line.t13)
                                          , tonumber(line.t14)
                                          , tonumber(line.t15)
                                          , tonumber(line.t16)
                                          , tonumber(line.t17)
                                          , tonumber(line.t18)
                                          , tonumber(line.t19)
                                          , tonumber(line.t20)
                                          , tonumber(line.t21)
                                          , tonumber(line.t22)
                                          , tonumber(line.t23)
                                          , tonumber(line.t24)
                                          , tonumber(line.t25)
                                          , tonumber(line.t26)
                                          , tonumber(line.t27)
                                         }
                                        )
            
            prediction_output = tmodel:forward(realtime_input)

            print(line.char_output)
            pprint(prediction_output) -- the output I'm printing looks very ugly. a thing to fix
            
        end -- end of validation condition

  end -- end of csv iterator
  



