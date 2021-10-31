# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import ssl
from flask_cors import CORS
from PIL import Image
import base64
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import cv2


app = Flask(__name__)
CORS(app)
isLocal = False


faceModel = tf.keras.models.load_model('face_model.h5')
faceModel.summary()

class_names=['CL (가수)', 'G.NA', '강동원', '강인 (가수)', '고아성', '공유', '구교환', '구하라', '권상우', '김건모', '김고은', '김민정', '김종국 (가수)', '김태희', '김혜수', '닉쿤', '류승범', '박진영', '박해일', '보아', '비 (가수)', '송강호', '송중기', '승리', '아이유', '안성기', '유노윤호', '유이 (배우)', '이병헌', '이승기', '이정재', '이효리', '장나라', '장동건', '전도연', '조성모', '최강창민', '최민식', '케이윌', '하정우', '한석규', '허각', '현아', '황정민', '효린']

def predictFace(imageList):
    
    resultList=[]
    
    for image in imageList:
        image_resized = cv2.resize(image,(100,100),interpolation=cv2.INTER_CUBIC)
        img_array = tf.expand_dims(image_resized, 0) # Create a batch

        predictions = faceModel.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        sortedScore = np.argsort(-score)[0:3]

        celebFaceList = []# 최종 3위 얼굴 리스트

        i = 0;        
        for rank in sortedScore:
            face={
                'name':class_names[rank],
                'percent': str(round((score[rank]*100).numpy(),2)),
                'num':str(rank)
            }
            print("[{}]순위 : {} - {:.2f}%".format(i+1,class_names[rank],score[rank]*100))
            celebFaceList.append(face)
            i+=1
        resultList.append(celebFaceList)
    return resultList

@app.route('/')
def test():
    return 'test'

@app.route('/faceRecog',methods=['POST'])
def face_recog():
    # f = request.files['file']
    # f.save(secure_filename(f.filename))

    # print(request.get_json())
    faceList = request.get_json()

    faceImageList = [];

    for face in faceList:
        base64_decoded = base64.b64decode(face['image'].split(',')[1])
        image = Image.open(io.BytesIO(base64_decoded))
        image_np = np.asarray(image)[:,:,:3] # 4채널 Transparency 삭제

        #image.show(Image.fromarray(image_np)) 디버깅용
        faceImageList.append(image_np)
        
    predictList = predictFace(faceImageList)

    print(predictList)

    return json.dumps(predictList)

if __name__ == "__main__":
    app.debug = True


    
    
    if isLocal:
        app.run(host="0.0.0.0", port=8575)
    else:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
        ssl_context.load_cert_chain(certfile='/etc/letsencrypt/live/minwoo.org/fullchain.pem', keyfile='/etc/letsencrypt/live/minwoo.org/privkey.pem')
        app.run(host="0.0.0.0", port=8575, ssl_context=ssl_context)