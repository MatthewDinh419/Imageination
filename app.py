from flask import Flask, render_template, request, redirect
app = Flask(__name__)
import os

app.config["IMAGE_UPLOADS"] = os.path.join('static','uploads')

@app.route('/')
def homepage():
    return render_template("homepage.html")

@app.route('/upload', methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            print(image)
            return render_template("homepage.html", filename=os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
    return render_template("homepage.html")

@app.route('/history', methods=["GET"])
def history():
    return render_template("history.html")

#start flask app
app.run(host="0.0.0.0", port=5000)