#!/bin/bash

# This script should be ran from your local pc
# It starts an instance with a zone and project id (all hardcoded)
# Then it asks for a project directory which must contain the following:
# 1. project_root/utils/setup_instance.sh - which runs venv and setups packages
# 2. project_root/rnnlm/experiment_local_runners/run_your_model.sh - which runs the model with its parameters
# 3. project_root/rnnlm/experiment_local_runners/start_experiment.sh - which calls the above of the two
# This script runs project_root/rnnlm/experiment_local_runners/start_experiment.sh

# RUN THIS WHEN TERMINAL PWD IS PROJECT_ROOT FOLDER!!!

PROJECT_ID="rnnlm-moshe-amit"
INSTANCE_NAME="instance-1"
ZONE="us-east1-c"

current_dir=$(pwd)

gcloud config set disable_prompts true
gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE
echo -e "\n"
gcloud compute instances list
echo -e "\n"
gcloud compute instances start $INSTANCE_NAME --zone=$ZONE
echo -e "\n"
gcloud compute instances list
echo -e "\n"

echo -e "copying $current_dir/rnnlm"
gcloud compute scp --recurse $current_dir/rnnlm $INSTANCE_NAME:remote_project_dir/Bsc-Final-Project

echo -e "copying $current_dir/main.py"
gcloud compute scp --recurse $current_dir/main.py $INSTANCE_NAME:remote_project_dir/Bsc-Final-Project

echo -e "starting experiment\n"
# The following line causes RAM overflow, we need a workaround
# gcloud compute ssh $INSTANCE_NAME --command="chmod +x ~/remote_project_dir/Bsc-Final-Project/rnnlm/experiment_local_runners/start_experiment.sh && ~/remote_project_dir/Bsc-Final-Project/rnnlm/experiment_local_runners/start_experiment.sh"
echo -e "Experiment done.\n"

echo -e "Copying results to local machine\n"
gcloud compute scp --recurse $INSTANCE_NAME:remote_project_dir/Bsc-Final-Project/results $current_dir

echo "Stop instance? [Y]/n?"
read stop_input

if [[ $stop_input != "n" && $stop_input != "N" ]]; then
    gcloud compute instances stop $INSTANCE_NAME
else
    echo -e "Instance NOT stopped. run gcloud compute instances stop $INSTANCE_NAME\n"
fi
gcloud compute instances list
echo -e "\n"