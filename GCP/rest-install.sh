#!/bin/sh
sudo apt-get update
sudo apt-get install -y python3 python3-pip git
sudo pip3 install flask
sudo pip3 install -U virtualenv
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
sudo pip3 install tensorflow==2.0.0a0
sudo pip3 install sklearn
sudo pip3 install tqdm
sudo apt-get install -y python3 python3-pip python3-pillow python3-openalpr python3-redis
git clone https://github.com/MatthewDinh419/Imageination.git
cd Imageination
python3 app.py
