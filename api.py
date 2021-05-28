from flask import Flask, jsonify, render_template, Response,  make_response,request
import copy
import numpy as np
import cv2
import base64


from Prediction import Prediction

app = Flask(__name__)
predictionImage = Prediction()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/live', methods=['GET'])
def liveAnalsis():
    predictionLive = Prediction()
    predictionLive.liveCamPredict()
    return "hola"

    
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        fs = request.files.get('snap')
        print(fs)
        if fs :

            img = cv2.imdecode(np.frombuffer(fs.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
            imgEmotions= predictionImage.emotionDetection(copy.deepcopy(img))
            imgObjects=  predictionImage.objectDetection(copy.deepcopy(img))
            imgFaces=   predictionImage.facialDetection(copy.deepcopy(img))
            imgBehavior= predictionImage.behaviorDetection(copy.deepcopy(img))

            _,imgEmotionsEncoded=cv2.imencode('.jpg', imgEmotions)
            _,imgObjectsEncoded=cv2.imencode('.jpg', imgObjects)
            _,imgFacesEncoded=cv2.imencode('.jpg', imgFaces)
            _,imgBehaviorEncoded=cv2.imencode('.jpg', imgBehavior)

            buf = jsonify({'img': base64.b64encode(imgEmotionsEncoded).decode('ascii'),'img2': base64.b64encode(imgObjectsEncoded).decode('ascii'),'img3': base64.b64encode(imgFacesEncoded).decode('ascii'),'img4': base64.b64encode(imgBehaviorEncoded).decode('ascii')})
            
            return buf
        else:
            print("holi")
            return 'App stopped'
    
    return 'Hello World!'



if __name__ == "__main__":
    app.run(debug=True)
