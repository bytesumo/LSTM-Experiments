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
                


  Example LSTM.

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
  - to set out the training OFFSET treshold. 
  - run the script using torch, not plain luajit. www.torch.ch
  
Note:

This program will spool your CSV records through an LSTM, set up in training mode until
 it hits a threshold you set using the OFFSET variable. 

After that OFFSET, it will stop training, and then run the LSTM in Test mode, printing the
network's predictions alongside the training target "output" values for inspection.

---------------------------------------------------------------------------------------

]]

-- print(whohow)

require 'nn'
require 'nngraph'
require 'csv'
require 'rnn'

LSTM = require 'LSTM.lua'

---------------------------------------------------------------------------------------
---- CONFIGURE csv file input source, and column aliases

local filename = "dp_10_train_DJI_min.csv"
local threshold = 100000                     -- train up to this offset, then predict.

local params ={}
params.header = true
params.separator = ","


 -- use this params list to map columns/headers in your csv file to our internal variable names. Note "output" is to be mapped to "trend" but we need to 
 -- do some preprocessing to convert it from a char binomial output to a binary output on two output nodes, i.e.: [1,0] or [0,1]
 
params.columns = {  -- we list possible header aliases, and map them to our stable var
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
                 
---------------------------------------------------------------------------------------
---- DEFINE the LSTM size, structure, and initialise

local input_size = 28
local hidden_size = 300
local output_size = 300

local mlp_input_size = output_size
local mlp_hidden_size = 300
local mlp_hidden_size2 = 300
local mlp_output_size = 2

-- BUILD the LSTM structure

network = {LSTM.create(input_size,hidden_size), LSTM.create(hidden_size,hidden_size), LSTM.create(hidden_size, output_size)}

criterion = nn.MSECriterion() 

-- INITITALISE it's internal states

local previous_state = {
  {torch.zeros(1, hidden_size), torch.zeros(1,hidden_size)}, -- was previously    torch.zeros(1,output_size)}
  {torch.zeros(1, hidden_size), torch.zeros(1,hidden_size)},
  {torch.zeros(1, output_size), torch.zeros(1,output_size)}
  }
local output = nil
local next_state = {}
local feed_input = nil

---------------------------------------------------------------------------------------
---- DEFINE the final MLP final learning layers

mlp = nn.Sequential();  -- make a multi-layer perceptron
mlp:add(nn.Linear(mlp_input_size, mlp_hidden_size))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(mlp_hidden_size, mlp_hidden_size2))  
mlp:add(nn.Tanh())
mlp:add(nn.Linear(mlp_hidden_size2, mlp_output_size))  

-- set out the learning criterion for the mlp layer
criterion = nn.MSECriterion() 
local mlp_inputdata = nil
local feed_output = nil

local prediction_output = nil

---------------------------------------------------------------------------------------
---- FEED THE NETWORK WITH VALUES

local offset = 0
-- Open file iterator:
local csv = require("csv")
local f = csv.open(filename, params)



for line in f:lines() do
  offset = offset + 1
  
        -- INPUT TENSOR from CSV
        -- use this line to create the tensor input to the rnn "feed_input"
        
              feed_input = torch.Tensor({{  line.percent_change    ---- note very carefully the TWO curly braces on this line. {{ }} is tensor(1,n) that we need
                                          , line.t1
                                          , line.t2
                                          , line.t3
                                          , line.t4
                                          , line.t5
                                          , line.t6
                                          , line.t7
                                          , line.t8
                                          , line.t9
                                          , line.t10
                                          , line.t11
                                          , line.t12
                                          , line.t13
                                          , line.t14
                                          , line.t15
                                          , line.t16
                                          , line.t17
                                          , line.t18
                                          , line.t19
                                          , line.t20
                                          , line.t21
                                          , line.t22
                                          , line.t23
                                          , line.t24
                                          , line.t25
                                          , line.t26
                                          , line.t27
                                         }}
                                       )
        --print(feed_input)
        -- construct training labels
        if line.char_output == "U" then        -- transform our char class [U,D] into two nodes of binary outputs
              feed_output = torch.Tensor({{1,0}})
        else
              feed_output = torch.Tensor({{0,1}})   
        end        
        
        -- forward pass, puts the inputs into the rnn, plus prev state
        local layer_input = {feed_input, table.unpack(previous_state[1])}
        --print ("this is the  layer_input")
        --print(layer_input)

    -- push the inputs through the rnn to get the output
      for l = 1, #network do 
                  -- forward the input          
            local layer_output = network[l]:forward(layer_input)
                  -- save output state for next iteration
            
            table.insert(next_state, layer_output)
                  -- extract hidden state from output
                  
            local layer_h = layer_output[2]
                  -- prepare next layer's input or set the output
                  
            if l < #network then
              layer_input = {layer_h, table.unpack(previous_state[l + 1])}
              
            else
              output = layer_h
            end
      end -- of LSTM layer
      
      mlp_inputdata = output

---------------------------------------------------------------------------------------
---- TRAIN THE NETWORK AGAINST OUTPUTS
-- Now we have the results of the LSTM module, we can build a normal MPL layer to train to the output

 ---[[
  if offset < threshold then
    
    -- feed it to the neural network and the criterion
    criterion:forward(mlp:forward(mlp_inputdata), feed_output)

    -- train over this example in 3 steps
    -- (1) zero the accumulation of the gradients
    mlp:zeroGradParameters()
    -- (2) accumulate gradients
    mlp:backward(mlp_inputdata, criterion:backward(mlp.output, feed_output))
    -- (3) update parameters with a 0.01 learning rate
    mlp:updateParameters(0.01)
  
  else
  
  -- the network is trained a tiny bit on the data up to "threshold" records, then I output predictions against the test values...
  
  prediction_output = mlp:forward(mlp_inputdata)

  print(line.char_output, prediction_output)
  end
  
  --]]
  --print(mlp_inputdata)
  
end -- of the file iterator