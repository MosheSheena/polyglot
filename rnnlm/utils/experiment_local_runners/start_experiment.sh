#!/bin/bash

chmod +x ~/remote_project_dir/Bsc-Final-Project/utils/setup_instance.sh
~/remote_project_dir/Bsc-Final-Project/utils/setup_instance.sh

chmod +x ~/remote_project_dir/Bsc-Final-Project/rnnlm/experiment_local_runners/run_fast_lstm.sh
source ~/tensorflow/bin/activate
~/remote_project_dir/Bsc-Final-Project/rnnlm/experiment_local_runners/run_fast_lstm.sh