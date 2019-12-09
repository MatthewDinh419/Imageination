#!/bin/sh
gcloud beta compute disks create rest-server-disk \
    --zone us-west1-a \
    --size 50GB \
    --image-project ubuntu-os-cloud \
    --image-family ubuntu-1904 \
    --type pd-standard
python flask-launch.py "project_id"
python redis-launch.py "project_id"