# Imageination

A website where a user will upload an image and then a caption that describes what is going on in the image will be returned.

The application utilizes flask, a python web framework, for server deployment as well as materialize for the front-end.

Components:

    -Redis Database
    -Key/Value Pairs
    -Virtual Machine deployment using GCP
    -Restful API
    -Tensorflow
Requirements to be installed:

    -Redis v3.3.11
    -Flask v1.1.1
    -Tensorflow v2.0
How to Run:

    1) Install the dependencies listed above
    2) Run app.py
    
Google Cloud Virtual Machine Instructions
    
    This will create virtual machine instances which will start a server for flask and redis.
    
    1) Authenticate using "gcloud auth login"
    2) Set project id using gcloud config set project "project_id"
    3) Change "project_id" to your project id in the file GCP/setup.sh
    4) chmod +x GCP/setup.sh
    5) ./GCP/setup.sh