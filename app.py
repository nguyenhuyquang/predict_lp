import os
from flask import Flask, render_template, request
from main import regMain

app = Flask(__name__)
UPLOAD_FOLDER = '/sample'


@app.route("/main")
def main():
    return render_template('index.html')


@app.route("/reg-plate", methods=['POST'])
def regPlate():
    file = request.files['file']
    path = os.path.join(app.root_path, 'sample/' + file.filename)
    file.save(path)
    plate = regMain('sample/' + file.filename)
    return render_template('index.html', plate=plate)

if __name__ == "__main__":
    app.debug = True
    app.run()
