from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from demo import *

app = Flask(__name__)

UPLOAD_FOLDER = r'static/Market-1501-v15.09.15/pytorch\query\\'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def introduction():
    return render_template('introduce.html')

@app.route('/demo')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/query', methods=['GET'])
def query():
    return render_template('query.html')

@app.route('/query', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        id = filename.split('_')[0]
        query_path ="%04d" % int(id) + "\\" + filename
        print(query_path)

        demo(query_path=r'static/Market-1501-v15.09.15/pytorch\query'+ "\\" + str(query_path))

        files = []
        num = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk('D:\KLTN\person-reid\static\show'):
            for file in f:
                if file.find('png') > -1:
                    files.append(file)
                    num.append(file.split(".")[0][-1])

        return render_template('query.html', filename=filename, video_files=files, num=num)

    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename="Market-1501-v15.09.15/pytorch/query/" + "%04d" % int(filename.split('_')[0]) + "/" + filename), code=301)

@app.route('/showquery')
def showquery():
    return render_template('showquery.html')

if __name__=='__main__':
    app.run(debug=True)