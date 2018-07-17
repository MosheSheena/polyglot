#!/bin/bash

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

gcloud compute scp --recurse $project_dir $INSTANCE_NAME:remote_project_dir
echo -e "\n"
gcloud compute ssh $INSTANCE_NAME --command="echo h"
echo -e "\n"

echo "Stop instance? [Y]/n?"
read stop_input

if [[ $stop_input != "n" && $stop_input != "N" ]]; then
    gcloud compute instances stop $INSTANCE_NAME
else
    echo -e "Instance NOT stopped. run gcloud compute instances stop $INSTANCE_NAME\n"
fi
gcloud compute instances list
echo -e "\n"