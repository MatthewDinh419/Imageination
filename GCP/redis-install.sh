#!/bin/sh
sudo apt update
sudo apt install -y redis-server
sudo echo "bind 0.0.0.0 ::1" | sudo tee /etc/redis/redis.conf
sudo systemctl restart redis-server