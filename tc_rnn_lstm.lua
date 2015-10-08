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

--[[ require libraries ]]--

local csv = require("csv")

require 'cltorch'
require 'clnn'
require 'rnn'
require 'pprint'
-- require 'nn'
-- require 'math'

--[[ configure csv reading]]--

local filename = "dp_30_train_all_stocks.csv"
-- local filename = "../train/1444307892/dp_30_train_all_stocks.csv"
local csvparams ={}

csvparams.header = true
csvparams.separator = ","

 -- timeseries data in csv note:
 -- use this params list to map columns/headers in your csv file to our internal variable names. 
 -- Note desired "output" is to be mapped to the field in the file called "trend" but we need to also
 -- do some preprocessing to convert it from a char binomial output to a number output on two output nodes, i.e.: [1,0] or [0,1]
 
 -- NOTE: the current feature generator outputs 246 seperate features.
 
 
csvparams.columns = {  -- we list possible header aliases, and map them to our stable var
                    tseries  = { names = {"TCFG", "symbol", "instrument", "id"}}, 
                    userdate = { names ={"UserDate", "Date", "date", "TimeStamp", "time", "timestamp"}},
                    value = { names = {"Value", "value", "Close", "close", "price", "bid" }},
                    percent_change  = { names = {"unit_per_change", "change", "percent_change", "relative_change"}},
                    char_output  = { names = {"trend"}},
                    dummy  = { names = {"dummy"}},
                    T1  = { names = {"T1"}},
                    T2  = { names = {"T2"}},
                    T3  = { names = {"T3"}},
                    T4  = { names = {"T4"}},
                    T5  = { names = {"T5"}},
                    T6  = { names = {"T6"}},
                    T7  = { names = {"T7"}},
                    T8  = { names = {"T8"}},
                    T9  = { names = {"T9"}},
                    T10  = { names = {"T10"}},
                    T11  = { names = {"T11"}},
                    T12  = { names = {"T12"}},
                    T13  = { names = {"T13"}},
                    T14  = { names = {"T14"}},
                    T15  = { names = {"T15"}},
                    T16  = { names = {"T16"}},
                    T17  = { names = {"T17"}},
                    T18  = { names = {"T18"}},
                    T19  = { names = {"T19"}},
                    T20  = { names = {"T20"}},
                    T21  = { names = {"T21"}},
                    T22  = { names = {"T22"}},
                    T23  = { names = {"T23"}},
                    T24  = { names = {"T24"}},
                    T25  = { names = {"T25"}},
                    T26  = { names = {"T26"}},
                    T27  = { names = {"T27"}},
                    T28 = { names = {"T28"}},
                  T29 = { names = {"T29"}},
                  T30 = { names = {"T30"}},
                  T31 = { names = {"T31"}},
                  T32 = { names = {"T32"}},
                  T33 = { names = {"T33"}},
                  T34 = { names = {"T34"}},
                  T35 = { names = {"T35"}},
                  Dummy7 = { names = {"Dummy7"}},
                  Ez1 = { names = {"Ez1"}},
                  Ez2 = { names = {"Ez2"}},
                  Ez3 = { names = {"Ez3"}},
                  Ez4 = { names = {"Ez4"}},
                  Ez5 = { names = {"Ez5"}},
                  Ez6 = { names = {"Ez6"}},
                  Ez7 = { names = {"Ez7"}},
                  Ez8 = { names = {"Ez8"}},
                  Ez9 = { names = {"Ez9"}},
                  Ez10 = { names = {"Ez10"}},
                  Ez11 = { names = {"Ez11"}},
                  Ez12 = { names = {"Ez12"}},
                  Ez13 = { names = {"Ez13"}},
                  Ez14 = { names = {"Ez14"}},
                  Ez15 = { names = {"Ez15"}},
                  Ez16 = { names = {"Ez16"}},
                  Ez17 = { names = {"Ez17"}},
                  Ez18 = { names = {"Ez18"}},
                  Ez19 = { names = {"Ez19"}},
                  Ez20 = { names = {"Ez20"}},
                  Ez21 = { names = {"Ez21"}},
                  Ez22 = { names = {"Ez22"}},
                  Ez23 = { names = {"Ez23"}},
                  Ez24 = { names = {"Ez24"}},
                  Ez25 = { names = {"Ez25"}},
                  Ez26 = { names = {"Ez26"}},
                  Ez27 = { names = {"Ez27"}},
                  Ez28 = { names = {"Ez28"}},
                  Ez29 = { names = {"Ez29"}},
                  Ez30 = { names = {"Ez30"}},
                  Ez31 = { names = {"Ez31"}},
                  Ez32 = { names = {"Ez32"}},
                  Ez33 = { names = {"Ez33"}},
                  Ez34 = { names = {"Ez34"}},
                  Ez35 = { names = {"Ez35"}},
                  Ez36 = { names = {"Ez36"}},
                  Ez37 = { names = {"Ez37"}},
                  Ez38 = { names = {"Ez38"}},
                  Ez39 = { names = {"Ez39"}},
                  Ez40 = { names = {"Ez40"}},
                  Ez41 = { names = {"Ez41"}},
                  Ez42 = { names = {"Ez42"}},
                  Ez43 = { names = {"Ez43"}},
                  Ez44 = { names = {"Ez44"}},
                  Ez45 = { names = {"Ez45"}},
                  Ez46 = { names = {"Ez46"}},
                  Ez47 = { names = {"Ez47"}},
                  Ez48 = { names = {"Ez48"}},
                  Ez49 = { names = {"Ez49"}},
                  Ez50 = { names = {"Ez50"}},
                  Ez51 = { names = {"Ez51"}},
                  Ez52 = { names = {"Ez52"}},
                  Ez53 = { names = {"Ez53"}},
                  Ez54 = { names = {"Ez54"}},
                  Ez55 = { names = {"Ez55"}},
                  Ez56 = { names = {"Ez56"}},
                  Ez57 = { names = {"Ez57"}}, --]]
                  Ez58 = { names = {"Ez58"}} 

                 }                 
                 


--[[ model definition params ]]--

local threshold = 99000  -- train up to this offset in the file, then predict.
local inputSize = 1 -- 94      -- the number features in the csv file to use as inputs 
local hiddenSize = 2000     -- the number of hidden nodes
-- local hiddenSize2 = 100     -- the second hidden layer
local outputSize = 2      -- the number of outputs representing the prediction targat. 
local dropoutProb = 0.6   -- a dropout probability, might help with generalisation
local rho = 2000            -- the timestep history we'll recurse back in time when we apply gradient learning
local batchSize = 10     -- the size of the episode/batch of sequenced timeseries we'll feed the rnn
local lr = 0.0002        -- the learning rate to apply to the weights


--[[ build up model definition ]]--

-- create a trend guessing model, "tmodel"
tmodel = nn.Sequential()                                               -- wrapping it all in Sequential brings forward / backward methods to all layers in one go
tmodel:add(nn.Sequencer(nn.Identity()))                                -- untransfomed input data

tmodel:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize, rho )))     -- will create a complex network of lstm cells that learns back rho timesteps
tmodel:add(nn.Sequencer(nn.Dropout(dropoutProb)))                      -- I'm sticking in a place to do dropout, not strictly needed I don't think

--tmodel:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize, rho)))      -- creating a second layer of lstm cells
--tmodel:add(nn.Sequencer(nn.Dropout(dropoutProb)))
 
tmodel:add(nn.Sequencer(nn.Linear(hiddenSize, outputSize)))           -- reduce the output back down to the output class nodes
tmodel:add(nn.Sequencer(nn.LogSoftMax()))                             -- apply the log soft max, as we're guessing classes.
                                                                      -- when used with criterion of ClassNLLCriterion, is effectively CrossEntropy

-- set criterion
                                                                    -- arrays indexed at 1, so target class to be like [1,2], [2,1] not [0,1],[1,0]
-- criterion = nn.SequencerCriterion(nn.MSECriterion())              -- if you want a regression, use mse
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())           -- if you want a classifier, use this. 
-- criterion = nn.CrossEntropyCriterion()                           -- potential alt for LogSoftMax / ClassNLLCriterion, but you pass it 1d weights (??)



--[[ set out learning functions ]]--

-- code borrowed from an example by fabio. thanks!
function gradientUpgrade(model, x, y, criterion, learningRate, i)
  model:zeroGradParameters()
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
local tsID = "XX"

for line in f:lines() do              -- the file iterator, iterates by lines in my csv raw data file
    offset = offset + 1               -- this is a global file offset, counts the rownumber of the raw file
    
                -- inputs to learn from --
                if line.tseries == tsID then 
                  batchcounter = batchcounter +1    -- this tracks the offset inside an episode (aka chunk of time, the max length of which is rho)
                else
                  batchcounter = 1
                end
                tsID = line.tseries
                  
                -- notes:
                -- We load up our lua table, row by row indexed 1 .. batchSize, and for each row, we insert a tensor of input records. 
                feed_input[batchcounter] = torch.Tensor(  
                                          {  -- insert inputs into standard training dataset table, indexed by this offset
                                            tonumber(line.percent_change)             
                                  --[[        , tonumber(line.T1)
                                          , tonumber(line.T2)
                                          , tonumber(line.T3)
                                          , tonumber(line.T4)
                                          , tonumber(line.T5)
                                          , tonumber(line.T6)
                                          , tonumber(line.T7)
                                          , tonumber(line.T8)
                                          , tonumber(line.T9)
                                          , tonumber(line.T10)
                                          , tonumber(line.T11)
                                          , tonumber(line.T12)
                                          , tonumber(line.T13)
                                          , tonumber(line.T14)
                                          , tonumber(line.T15)
                                          , tonumber(line.T16)
                                          , tonumber(line.T17)
                                          , tonumber(line.T18)
                                          , tonumber(line.T19)
                                          , tonumber(line.T20)
                                          , tonumber(line.T21)
                                          , tonumber(line.T22)
                                          , tonumber(line.T23)
                                          , tonumber(line.T24)
                                          , tonumber(line.T25)
                                          , tonumber(line.T26)
                                          , tonumber(line.T27)
, tonumber(line.T28)
, tonumber(line.T29)
, tonumber(line.T30)
, tonumber(line.T31)
, tonumber(line.T32)
, tonumber(line.T33)
, tonumber(line.T34)
, tonumber(line.T35) 
, tonumber(line.Ez1)
, tonumber(line.Ez2)
, tonumber(line.Ez3)
, tonumber(line.Ez4)
, tonumber(line.Ez5)
, tonumber(line.Ez6)
, tonumber(line.Ez7)
, tonumber(line.Ez8)
, tonumber(line.Ez9)
, tonumber(line.Ez10) 
, tonumber(line.Ez11)
, tonumber(line.Ez12)
, tonumber(line.Ez13)
, tonumber(line.Ez14)
, tonumber(line.Ez15) 
, tonumber(line.Ez16)
, tonumber(line.Ez17)
, tonumber(line.Ez18)
, tonumber(line.Ez19)
, tonumber(line.Ez20)
, tonumber(line.Ez21)
, tonumber(line.Ez22)
, tonumber(line.Ez23)
, tonumber(line.Ez24)
, tonumber(line.Ez25)
, tonumber(line.Ez26)
, tonumber(line.Ez27)
, tonumber(line.Ez28)
, tonumber(line.Ez29)
, tonumber(line.Ez30) 
, tonumber(line.Ez31)
, tonumber(line.Ez32)
, tonumber(line.Ez33)
, tonumber(line.Ez34)
, tonumber(line.Ez35) 
, tonumber(line.Ez36)
, tonumber(line.Ez37)
, tonumber(line.Ez38)
, tonumber(line.Ez39)
, tonumber(line.Ez40)
, tonumber(line.Ez41)
, tonumber(line.Ez42)
, tonumber(line.Ez43)
, tonumber(line.Ez44)
, tonumber(line.Ez45)
, tonumber(line.Ez46)
, tonumber(line.Ez47)
, tonumber(line.Ez48)
, tonumber(line.Ez49)
, tonumber(line.Ez50)
, tonumber(line.Ez51)
, tonumber(line.Ez52)
, tonumber(line.Ez53)
, tonumber(line.Ez54)
, tonumber(line.Ez55)
, tonumber(line.Ez56)
, tonumber(line.Ez57) 
, tonumber(line.Ez58) 

                                          --]]
                                         }
                                        )
                 
                --[[ targets to learn ]]--
                
                -- construct flags that map to our training labels. (Remember classes indexed from 1 .. n), so our
                -- char class of [U,D] becomes {{2,1},{1,2}}, or as Balint pointed out, {{1},{2}}
                
                if line.char_output == "D"  then   -- "D"                        
                      feed_output[batchcounter] = torch.FloatTensor({1})    -- set Down as 1
                else   -- else must be "U" "1"                                               
                      feed_output[batchcounter] = torch.FloatTensor({2})    -- set Up as 2
                end   
                
        --[[ TRAIN THE RNN ]]-- (note we are still inside the file iterator here)

        if offset < threshold and offset % batchSize == 0 then    -- bactSize defines the episode size
          episode = episode + 1                                   -- create a counter that tracks the total number of episodes we've created
          
          -- now send this batch episode to rnn backprop through time learning
         
          -- to use the cross entropy criterion, I need to hack my targets to just a 1d tensor
          -- a better way is include a LogSoftMax layer with 
          -- feed_output = feed_output:view(-1)
          --local feed_target = feed_output[batchounter]:view(-1)

              gradientUpgrade(tmodel, feed_input, feed_output, criterion, lr, episode)
          
          -- now clear out the tables, reset them     
          feed_input = nil; feed_input = {}
          feed_output = nil; feed_output = {}
          feed_offsets = nil; feed_offsets = {}
          
          -- reset the rowID of the batch episode back to zero
          batchcounter = 0
          
        end -- end of the batchSize full event trigger 
        
        --[[ Validation ]]--
        
        if offset > threshold and offset < 133900 then        -- we have now rolled through the timeseries learning, but can we guess the ending?   
                                          -- note we are still inside the file iterator. 
            -- TEST OF PREDICTION --            
            -- grab the current row of inputs, and generate prediction
            
            local realtime_input = {}
            realtime_input[1] = torch.Tensor(  
                                          {  -- insert inputs into standard training dataset table, indexed by this offset
                                          --  tonumber(line.percent_change)          
                                          --, 
                                            tonumber(line.percent_change)             
                                  --[[        , tonumber(line.T1)
                                          , tonumber(line.T2)
                                          , tonumber(line.T3)
                                          , tonumber(line.T4)
                                          , tonumber(line.T5)
                                          , tonumber(line.T6)
                                          , tonumber(line.T7)
                                          , tonumber(line.T8)
                                          , tonumber(line.T9)
                                          , tonumber(line.T10)
                                          , tonumber(line.T11)
                                          , tonumber(line.T12)
                                          , tonumber(line.T13)
                                          , tonumber(line.T14)
                                          , tonumber(line.T15)
                                          , tonumber(line.T16)
                                          , tonumber(line.T17)
                                          , tonumber(line.T18)
                                          , tonumber(line.T19)
                                          , tonumber(line.T20)
                                          , tonumber(line.T21)
                                          , tonumber(line.T22)
                                          , tonumber(line.T23)
                                          , tonumber(line.T24)
                                          , tonumber(line.T25)
                                          , tonumber(line.T26)
                                          , tonumber(line.T27)
, tonumber(line.T28)
, tonumber(line.T29)
, tonumber(line.T30)
, tonumber(line.T31)
, tonumber(line.T32)
, tonumber(line.T33)
, tonumber(line.T34)
, tonumber(line.T35) 
, tonumber(line.Ez1)
, tonumber(line.Ez2)
, tonumber(line.Ez3)
, tonumber(line.Ez4)
, tonumber(line.Ez5)
, tonumber(line.Ez6)
, tonumber(line.Ez7)
, tonumber(line.Ez8)
, tonumber(line.Ez9)
, tonumber(line.Ez10) 
, tonumber(line.Ez11)
, tonumber(line.Ez12)
, tonumber(line.Ez13)
, tonumber(line.Ez14)
, tonumber(line.Ez15) 
, tonumber(line.Ez16)
, tonumber(line.Ez17)
, tonumber(line.Ez18)
, tonumber(line.Ez19)
, tonumber(line.Ez20)
, tonumber(line.Ez21)
, tonumber(line.Ez22)
, tonumber(line.Ez23)
, tonumber(line.Ez24)
, tonumber(line.Ez25)
, tonumber(line.Ez26)
, tonumber(line.Ez27)
, tonumber(line.Ez28)
, tonumber(line.Ez29)
, tonumber(line.Ez30) 
, tonumber(line.Ez31)
, tonumber(line.Ez32)
, tonumber(line.Ez33)
, tonumber(line.Ez34)
, tonumber(line.Ez35) 
, tonumber(line.Ez36)
, tonumber(line.Ez37)
, tonumber(line.Ez38)
, tonumber(line.Ez39)
, tonumber(line.Ez40)
, tonumber(line.Ez41)
, tonumber(line.Ez42)
, tonumber(line.Ez43)
, tonumber(line.Ez44)
, tonumber(line.Ez45)
, tonumber(line.Ez46)
, tonumber(line.Ez47)
, tonumber(line.Ez48)
, tonumber(line.Ez49)
, tonumber(line.Ez50)
, tonumber(line.Ez51)
, tonumber(line.Ez52)
, tonumber(line.Ez53)
, tonumber(line.Ez54)
, tonumber(line.Ez55)
, tonumber(line.Ez56)
, tonumber(line.Ez57) 
, tonumber(line.Ez58) --]]
                                         }
                                        )
            
            prediction_output = torch.Tensor()
            prediction_output = tmodel:forward(realtime_input)

            --print(line.char_output)
            --pprint(prediction_output[1]:exp())
            local classProbabilities = {}
            local classPrediction = ""
            classProbabilities = torch.totable(prediction_output[1]:exp())
            
            
            -- looking at my results I'm a little worried that I'm getting the guesses backwards! aargh.
            if classProbabilities[1] < classProbabilities[2] then
                 classPrediction = "Up"
            else classPrediction = "Down"
            end
              
            local delim =","
            
            print( -- line.tseries           .. delim ..
                   -- line.userdate          .. delim ..
                   line.char_output       .. delim .. 
                   -- classProbabilities[1]  .. delim .. 
                   -- classProbabilities[2]  .. delim .. 
                   classPrediction        .. delim ..
                   line.value             
                  )
            
            -- the output I'm printing looks very ugly. a thing to fix
            -- pprint(prediction_output) 
                    --[[
                    tseries  = { names = {"TCFG", "symbol", "instrument", "id"}}, 
                    userdate = { names ={"UserDate", "Date", "date", "TimeStamp", "time", "timestamp"}},
                    value = { names = {"Value", "value", "Close", "close", "price", "bid" }},
                    percent_change  = { names = {"unit_per_change", "change", "percent_change", "relative_change"}},
                    char_output  = { names = {"trend"}},
                    --]]
            
            
            
            
        end -- end of validation condition

  end -- end of csv iterator
  



