from flask import Flask, jsonify, render_template, Response,  make_response, request, redirect, url_for,send_file
import copy
import numpy as np
import cv2
import base64
import glob
import os
from datetime import datetime
from Prediction import Prediction

app = Flask(__name__,static_folder='static')
predictionImage = Prediction()

released = False

frames = []


@app.route('/')
def index():
    predictionImage = Prediction()
    return render_template('index.html')

@app.route('/videos')
def videos():
    for current_dir, dirs, files in os.walk('static/videos/'):
        print(files)

    return render_template('videos.html', videos=files)

@app.route('/reports')
def reports():
    for current_dir, dirs, files in os.walk('static/reports/'):
        print(files)
    return render_template('reports.html', reports=files)

@app.route("/download/<string:folder>/<string:subfolder>/<string:file>", methods=['GET'])
def download (folder = None, subfolder = None, file= None):
    route = [folder,subfolder,file]
    print("PATH! ", '/'.join(route))
    if None in route:    
        return "Elemento no encontrado"
    try:
        return send_file('/'.join(route), as_attachment=True)
    except Exception as e:
        return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global released, frames

    if request.method == 'POST':
        fs = request.files.get('snap')
        if fs:
            img = cv2.imdecode(np.frombuffer(
                fs.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            imgEmotions = predictionImage.emotionDetection(copy.deepcopy(img))
            imgObjects = predictionImage.objectDetection(copy.deepcopy(img))
            imgFaces = predictionImage.facialDetection(copy.deepcopy(img))
            imgBehavior = copy.deepcopy(img)#predictionImage.behaviorDetection(copy.deepcopy(img))

            cv2.resize(imgEmotions, (480, 480),interpolation=cv2.INTER_CUBIC)
            cv2.resize(imgObjects, (480, 480),interpolation=cv2.INTER_CUBIC)
            cv2.resize(imgFaces, (480, 480),interpolation=cv2.INTER_CUBIC)
            cv2.resize(imgBehavior, (480, 480),interpolation=cv2.INTER_CUBIC)

            imgEmotions = np.array(imgEmotions)
            imgEmotions = imgEmotions.astype('uint8')
            imgObjects = np.array(imgObjects)
            imgObjects = imgObjects.astype('uint8')
            imgFaces = np.array(imgFaces)
            imgFaces = imgFaces.astype('uint8')
            imgBehavior = np.array(imgBehavior)
            imgBehavior = imgBehavior.astype('uint8')

            vid1 = cv2.hconcat([imgEmotions,imgObjects ])
            vid2 = cv2.hconcat([imgFaces,imgBehavior ])
            
            frame = cv2.vconcat([vid1, vid2])

            cv2.resize(frame, (1280, 720))

            frames.append(frame)

            _, imgEmotionsEncoded = cv2.imencode('.jpg', imgEmotions)
            _, imgObjectsEncoded = cv2.imencode('.jpg', imgObjects)
            _, imgFacesEncoded = cv2.imencode('.jpg', imgFaces)
            _, imgBehaviorEncoded = cv2.imencode('.jpg', imgBehavior)
            
            predictionImage.frames+=1

            buf = jsonify({'img': base64.b64encode(imgEmotionsEncoded).decode('ascii'), 'img2': base64.b64encode(imgObjectsEncoded).decode(
                'ascii'), 'img3': base64.b64encode(imgFacesEncoded).decode('ascii'), 'img4': base64.b64encode(imgBehaviorEncoded).decode('ascii')})

            released = False
            return buf
        else:
            if (released == False):
                makeVideo()
                
            return 'App stopped'

    return 'Hello World!'

def makeVideo():
    global frames

    height, width, _ = frames[0].shape
    size = (width,height)
    
    out = cv2.VideoWriter(f'static/videos/project_{str(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 3, size)
    
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
    
    predictionImage.makeReport()
    frames=[]
    predictionImage.frames=0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(debug=True)
