#!/bin/bash

cd ~/remote_project_dir/Bsc-Final-Project

python ./src/models/lstm_fast.py --data-path=./data --vocab-path=./data/wordlist.rnn.final --save-path=./results
echo "Done"
