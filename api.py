from flask import Flask, jsonify, render_template, Response,  make_response,request
import datetime
import numpy as np
import cv2

from Prediction import Prediction

app = Flask(__name__)
predictionImage = Prediction()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/vid/<string:route>', methods=['GET'])
def videoAnalsis(route):
    predictionVideo = Prediction()
    predictionVideo.videoPrediction(route)
    return "holaa"



@app.route('/api/img/<string:route>', methods=['GET'])
def imageAnalsis(route):
    predictionImage = Prediction()
    predictionImage.imagePrediction(route, None, 0)
    return "hola"


@app.route('/api/live', methods=['GET'])
def liveAnalsis():
    predictionLive = Prediction()
    predictionLive.liveCamPredict()
    return "hola"


@app.route('/api/summary', methods=['GET'])
def summary():
    predictionSummary = Prediction()
    predictionSummary.showAccuracy()
    

@app.route('/api/graphic', methods=['GET'])
def graphic():
    predictionGraphic = Prediction()
    return predictionGraphic.trainingGraphics()

def send_file_data(data, mimetype='image/jpeg', filename='output.jpg'):
    
    response = make_response(data)
    response.headers.set('Content-Type', mimetype)
    response.headers.set('Content-Disposition', 'attachment', filename=filename)
    
    return response
    
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        fs = request.files.get('snap')
        if fs:

            img = cv2.imdecode(np.frombuffer(fs.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            img=predictionImage.imagePrediction("", img, 0)
            
            text = datetime.datetime.now().strftime('%Y.%m.%d %H.%M.%S.%f')
            img = cv2.putText(img, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 

            ret, buf = cv2.imencode('.jpg', img)
            
            return send_file_data(buf.tobytes())
        else:
            return 'You forgot Snap!'
    
    return 'Hello World!'



if __name__ == "__main__":
    app.run(debug=True)
