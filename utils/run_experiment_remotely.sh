#!/bin/bash

# This script should be ran from your local pc
# It starts an instance with a zone and project id (all hardcoded)
# Then it asks for a project directory which must contain the following:
# 1. project_root/utils/setup_instance.sh - which runs venv and setups packages
# 2. project_root/src/experiment_local_runners/run_your_model.sh - which runs the model with its parameters
# 3. project_root/src/experiment_local_runners/start_experiment.sh - which calls the above of the two
# This script runs project_root/src/experiment_local_runners/start_experiment.sh

PROJECT_ID="rnnlm-moshe-amit"
INSTANCE_NAME="instance-1"
ZONE="us-east1-c"

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

echo "Enter full path for project directory:"

read project_dir

echo -e "copying $project_dir/data"
gcloud compute scp --recurse $project_dir/data $INSTANCE_NAME:remote_project_dir/Bsc-Final-Project
echo -e "copying $project_dir/src"
gcloud compute scp --recurse $project_dir/src $INSTANCE_NAME:remote_project_dir/Bsc-Final-Project
echo -e "copying $project_dir/utils"
gcloud compute scp --recurse $project_dir/utils $INSTANCE_NAME:remote_project_dir/Bsc-Final-Project

echo -e "starting experiment\n"
# The following line causes RAM overflow, we need a workaround
# gcloud compute ssh $INSTANCE_NAME --command="chmod +x ~/remote_project_dir/Bsc-Final-Project/src/experiment_local_runners/start_experiment.sh && ~/remote_project_dir/Bsc-Final-Project/src/experiment_local_runners/start_experiment.sh"
echo -e "Experiment done.\n"

echo -e "Copying results to local machine\n"
gcloud compute scp --recurse $INSTANCE_NAME:remote_project_dir/$project_dir/results $project_dir

echo "Stop instance? [Y]/n?"
read stop_input

if [[ $stop_input != "n" && $stop_input != "N" ]]; then
    gcloud compute instances stop $INSTANCE_NAME
else
    echo -e "Instance NOT stopped. run gcloud compute instances stop $INSTANCE_NAME\n"
fi
gcloud compute instances list
echo -e "\n"