#!/bin/sh
sudo apt-get update
sudo apt-get install -y python3 python3-pip git
sudo pip3 install flask
sudo apt-get install -y python3 python3-pip python3-pillow python3-openalpr python3-redis
git clone https://github.com/MatthewDinh419/Imageination.git
cd Imageination
python3 app.py



