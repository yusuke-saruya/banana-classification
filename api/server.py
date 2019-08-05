from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import os
import string
import random
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import decode_predictions

# 美味しいバナナかどうかを判定するモデル
model = keras.models.load_model('../banana2.h5')
model._make_predict_function()
graph = tf.get_default_graph()

category = np.array([
    '未熟なバナナだ。もう少し待とう！',
    '良いバナナだ！ いますぐ食べよう！',
    '危険なバナナだ！ 捨てるか、覚悟して食べよう！',
    'これはバナナではない！'
    ])

SAVE_DIR = "./images"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

message = 'バナナの画像を選んで送信してください!'

app = Flask(__name__, static_url_path="")

def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])

@app.route('/')
def index():
    return render_template(
            'index.html',
            message=message
            )

@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory(SAVE_DIR, path)

# 参考: https://qiita.com/yuuuu3/items/6e4206fdc8c83747544b
@app.route('/upload', methods=['POST'])
def upload():
    if request.files['image']:

        # 画像として読み込み
        stream = request.files['image'].stream

        # ファイル名を読み込み
        filename = (request.files['image']).filename
        original_path = './images/' + filename

        # model読み込み用に画像ファイルを変換
        img = image.load_img(stream, target_size=(224, 224))
        x = image.img_to_array(img)
        #一回保存
        image.save_img(original_path, x)
        x = np.expand_dims(x, axis=0)

        # これをしないとpredict時にエラーになる
        # https://github.com/keras-team/keras/issues/10431
        global graph
        with graph.as_default():
            preds = model.predict(x)

        message = category[np.argmax(preds)]

        print(message)

    return render_template(
            'index.html',
            message=message,
            filename=original_path
            )

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8888)
