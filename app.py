from flask import Flask, render_template, request, redirect
import os
import commons
import text_sentiments
from PIL import Image
import uuid

app = Flask(__name__)
port = 5000

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if len(os.listdir('static/image')) != 0:
        for f in os.listdir('static/image'):
            os.remove(os.path.join('static/image', f))
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html')
        file = request.files.get('file')
        text = request.form['text']
        text_model = request.form['text_model']
        image_model = request.form['image_model']
        if not file:
            return render_template('index.html')
        txt, proba_txt, image_mood=text_sentiments.detect_mood_roberta(text,text_model)
        img=Image.open(file)
        image_name = str(uuid.uuid4())
        image_name = 'static/image/'+image_name + '.jpg'
        img.save(image_name)
        img_prediction, img_proba = commons.get_prediction(img, image_model)
        return render_template('result.html', img_proba=img_prediction,img_prediction=img_proba, mood=txt, image_mood=image_mood, proba_txt=proba_txt, image_name=image_name)
    return render_template('index.html')



