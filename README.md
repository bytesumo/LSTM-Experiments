# LSTM-Experiments

Torch7 LSTM Experiment


My experiments hacking around with lstm for torch7.

My code is based on the code examples and article published here:
http://apaszke.github.io/lstm-explained.html


Current experiment design:

a. stream in csv timeseries data (don't read it all in at once)
b. feed this data into the example lstm, row at a time
c. push the lstm output into a little feed forward net
d. do some training - to make it simple, train on the first x rows of the timeseries, then guess the remainder
e. print the predictions, alongside the actual to get a sense of whether it worked

to run:

>th tc_lstm.lua

depends on an installation of torch7 and the csv library found here:
https://github.com/geoffleyland/lua-csv
