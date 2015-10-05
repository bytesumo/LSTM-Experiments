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
                    -- train up to this offset, then predict.
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


local threshold = 100000 
local inputSize = 28
local hiddenSize = 50
local outputSize = 2
local dropoutProb = 0.2
local rho = 50
local batchSize = 200
local lr = 0.001



--[[ build up model definition ]]--

---[[ -- ALL OF THE COMMENT OUT SECTION THROWS AN ERROR ARISING IN FastLSTM

tmodel = nn.Sequential()
tmodel:add(nn.Sequencer(nn.Identity()))

tmodel:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize, rho))) ----update rnn package was fix for failures on this call
tmodel:add(nn.Sequencer(nn.Dropout(dropoutProb)))

tmodel:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize, rho))) 

tmodel:add(nn.Sequencer(nn.Linear(hiddenSize, outputSize)))
tmodel:add(nn.Sequencer(nn.LogSoftMax()))


--criterion = nn.SequencerCriterion(nn.MSECriterion())      -- if you want a regression, use mse
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())   -- if you want a classifier, use this. Outputs must be like [1,2] or [2,1] etc not [0,1][1,0]
--[[ set out learning functions ]]--


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

-- Open file iterator:
local f = csv.open(filename, csvparams)

-- create structured lua tables from csv. for the moment they are global till I figure stuff out

feed_input = {}     -- create a table to hold our input data we'll use to predict, I want a table of tensors
feed_output = {}    -- create a table to hold our target outputs to train against
feed_offsets = {}   -- create a table to hold our batch offset references -- not sure I need this

for line in f:lines() do
    offset = offset + 1               -- this is a global file offset
    batchcounter = batchcounter +1    -- this tracks the offset inside an episode (aka chunk of time, the max length of which is rho)
    -- We load up our data row by row into a table, then after 100 rows, we flush to tensor, and feed it to the training as a mini-batch
                -- this needs changing. My input needs to be a tensor for each line
                -- and then I collect up a table of tensors indexed by offset until I hit rho. Do this for both input and targets (I think)
               
                --feed_input[offset] = 
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
                 
                 -- print(feed_input[offset])
                                      -- )
                -- construct training labels to learn
                if line.char_output == "U" then        -- transform our char class [U,D] into two nodes of numeric [1,1] outputs
                      feed_output[batchcounter] = torch.Tensor({2, 1})  --  torch.Tensor({2,1}) 
                else -- then line.chart_output must be "D"
                      feed_output[batchcounter] = torch.Tensor({1, 2}) -- torch.Tensor({1,2}) , remember classes indexed from 1 .. n  
                end   
                
                -- create an index of the offsets, needed for batch processing
                -- table.insert(feed_offsets, batchcounter)
                -- offset = 0
        
        --[[ TRAIN THE RNN ]]--


        if offset < threshold and offset % batchSize == 0 then -- rho defines the episode size
          episode = episode + 1                          -- episode counter tracks the total number of episodes we've created
          
          -- now send the batch to learning
         
          gradientUpgrade(tmodel, feed_input, feed_output, criterion, lr, episode)
          
          -- now clear out the tables, reset them     
          feed_input = nil; feed_input = {}
          feed_output = nil; feed_output = {}
          feed_offsets = nil; feed_offsets = {}
          batchcounter = 0
          
          -- I delete these down as the next loop will rebuild these up to the next offset % 100 batch...
        end -- of the rho episode-sized block condition
        
        if offset > threshold then
          
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
                                        
            -- if realtime_input:size() == 28 then
            
            prediction_output = tmodel:forward(realtime_input)
            
            -- end
            --local output = tensor.totable(prediction_output[1])
            --local guess = ""
            --if output[1] > output[1]
            --  then guess = "up"
            --else guess = "down"
            --end

            print(line.char_output, "< the target")
            pprint(prediction_output)
            
        end

  end -- end of csv iterator
  

