#!/usr/bin/python3
from flask import Flask, render_template, request
import os
import redis

app = Flask(__name__)

#Setup base directory for images
app.config["IMAGE_UPLOADS"] = os.path.join('static','uploads')

#Train Model
import train

#Setup redis database
hostname = "redis-server"
redis_history = redis.Redis(host=hostname, db=1)

@app.route('/')
def homepage():
    return render_template("homepage.html", filename_dict=None)

@app.route('/upload', methods=["POST"])
def upload():
    if request.files:
        image = request.files["image"]
        img_path = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
        image.save(img_path)

        print("try")
        result, attention_plot = train.evaluate(img_path)
        print('Prediction Caption:', ' '.join(result[:-1]))

        caption = ' '.join(result[:-1])
        redis_history.set(image.filename, caption)
        filename_caption_dict = {}
        filename_caption_dict[image.filename] = caption

        return render_template("homepage.html", filename_dict=filename_caption_dict)
    return render_template("homepage.html")

@app.route('/history', methods=["GET"])
def history():
    filename_caption_dict = {}
    for key in redis_history.keys("*"):
        print(key.decode('utf-8'))
        print(redis_history.get(key))
        print("========================")
        filename_caption_dict[key.decode('utf-8')] = redis_history.get(key).decode('utf-8')

    return render_template("history.html", redis_json=filename_caption_dict)

#start flask app
app.run(host="0.0.0.0", port=5000)