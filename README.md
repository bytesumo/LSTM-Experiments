# LSTM-Experiments

Torch7 LSTM Experiment


My experiments hacking around with lstm for torch7.

My code was inspired by a great article published here:
http://apaszke.github.io/lstm-explained.html

NOTES:

I moved on to the rnn library as I got very muddled trying to figure out how to train my own lstm.

This code still not managing to learn properly. I'm getting an issues with a nil gradient error.
I'm still working on this - if anyone figures out the bug, let me know





Current experiment design:

a. stream in csv timeseries data (don't read it all in at once, my files bigger than ram)

b. feed this data into the example lstm. every rho steps, build a table of tensors and feed it to learning...

d. do some training - to make it simple, train on the first x rows of the timeseries, then guess the remainder

e. print the predictions, alongside the actual to get a sense of whether it worked

to run:

>th tc_rnn_lstm.lua 

depends on an installation of torch7 and the csv library found here:
https://github.com/geoffleyland/lua-csv
