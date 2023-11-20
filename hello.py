from flask import Flask
from flask import render_template
from distutils.log import debug
from fileinput import filename
from flask import *
from appprocess import preprocess, rearrange, w2v, get_mean_vector, avgFeatureVector, rfc, identify 
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
 
app.secret_key = 'This is your secret key to utilize session in Flask'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
          # upload file flask
        raws = [request.form['news']]
        if len(raws[0]) != 0:
            news = preprocess(raws)
            fixed = rearrange(news)

            model = w2v()
            vect = avgFeatureVector(news, model)

            model_rf = rfc()
            result = identify(model_rf, vect)
            
            if result[0] == 1:
                return render_template('verify2.html', color_news='green', data_news=raws[0], data_result='Valid')
            else:
                return render_template('verify2.html', color_news='red', data_news=raws[0], data_result='Hoax')

        else:
            raws=['Teks berita tidak boleh kosong']

            return render_template('verify2.html', color_news='red', data_news=raws[0], data_result='Error')

    return render_template('verify.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)