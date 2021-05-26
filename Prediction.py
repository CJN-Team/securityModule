import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from humanfriendly import format_timespan
import base64
from io import BytesIO
import os
import pickle
import face_recognition


class Prediction:

    def __init__(self):
        self.loadJsonModel('models/emotions/fer.json')
        self.initCV2()
        self.emotions = ['anger', 'disgust', 'fear',
                         'happiness', 'neutral', 'sadness', 'surprise']
        self.names = ['mobile phone', 'person']
        with open('models/emotions/modelParams.txt') as json_file:
            self.data = json.load(json_file)
        self.report = []
        self.previous = ["", 0, 0.0]
        self.classNames = self.loadNames()
        self.encodeListKnown = self.encode()

    def loadJsonModel(self, route):

        json_file = open(route, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.emotionsModel = model_from_json(loaded_model_json)
        self.emotionsModel.load_weights('models/emotions/fer.h5')

        json_file = open('models/objects/examanager.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.objectModel = model_from_json(loaded_model_json)
        self.objectModel.load_weights('models/objects/yolov3-examanager_weights.tf')

    def initCV2(self):
        self.face_haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.cap = cv2.VideoCapture(0)

    def liveCamPredict(self, img):
        cap = cv2.VideoCapture(0)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
        frames = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        while(cap.isOpened()):

            ret, img = cap.read()

            if ret == True:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05,
                                                                         minNeighbors=5, minSize=(30, 30),
                                                                         flags=cv2.CASCADE_SCALE_IMAGE)

                for (x, y, w, h) in faces_detected:

                    cv2.rectangle(img, (x, y), (x + w, y + h),
                                  (0, 255, 0), thickness=2)
                    roi_gray = gray_img[y:y + w, x:x + h]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    img_pixels = image.img_to_array(roi_gray)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255.0

                    predictions = self.emotionsModel.predict(img_pixels)
                    max_index = int(np.argmax(predictions))

                    predicted_emotion = self.emotions[max_index]

                    cv2.putText(img, predicted_emotion + " "+str(predictions[0][max_index]), (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    resized_img = cv2.resize(img, (1000, 700))
                    #cv2.imshow('Facial Emotion Recognition', resized_img)

                    self.flagCreation(predictions, max_index,
                                      predicted_emotion, frames)

                out.write(img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

            if(len(self.report) > 0):
                if(isinstance(self.report[-1][2], int)):
                    self.report[-1][2] = self.timeConversion(
                        (self.report[-1][2]/fps))
            frames += 1
        self.makeReport()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def imagePrediction(self, route, imageCaputured, frames):

        img = None
        if route == "":
            img = imageCaputured
        else:
            img = cv2.imread(route)

        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
        
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(self.encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(self.encodeListKnown,encodeFace)
            matchIndex = np.argmin(faceDis)
        
            if faceDis[matchIndex]< 0.50:
                
                name = self.classNames[matchIndex].upper()
                name = name.split('_')[0]

            else: name = 'Unknown'
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05,
                                                                 minNeighbors=5, minSize=(30, 30),
                                                                 flags=cv2.CASCADE_SCALE_IMAGE)
        img_pixels = image.img_to_array(img)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        objectPrediction = self.objectModel.predict(
            tf.image.resize(img_pixels, [416, 416]))

        boxes, scores, classes, nums = self.output_boxes( \
            objectPrediction, (413,413,3),
            max_output_size=40,
            max_output_size_per_class=20,
            iou_threshold=0.5,
            confidence_threshold=0.3)

        imagobj = np.squeeze(img_pixels)
        img = self.draw_outputs(imagobj, boxes, scores, classes, nums, self.names)

        for (x, y, w, h) in faces_detected:
            
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0

            predictions = self.emotionsModel.predict(img_pixels)

            max_index = int(np.argmax(predictions))

            predicted_emotion = self.emotions[max_index]
            cv2.putText(img, predicted_emotion + " "+str(predictions[0][max_index]), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1)
            resized_img = cv2.resize(img, (1000, 700))
            self.flagCreation(predictions, max_index,
                              predicted_emotion, frames)

        
        """    if route != "":
                cv2.imshow('Facial Emotion Recognition', resized_img)
        if route != "":
            cv2.waitKey(0)
        """
        return img
        # self.cleaningCV2()

    def loadNames(self):
        path = 'dataset'
        images = []
        classNames = []
        myList = os.listdir(path)

        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])

        return classNames

    def encode(self):
        with open('dataset_faces.dat', 'rb') as f:
            encodeListKnown = pickle.load(f)
        return encodeListKnown

    def output_boxes(self,inputs, model_size, max_output_size, max_output_size_per_class,
                     iou_threshold, confidence_threshold):
        center_x, center_y, width, height, confidence, classes = tf.split(
            inputs, [1, 1, 1, 1, 1, -1], axis=-1)
        top_left_x = center_x - width / 2.0
        top_left_y = center_y - height / 2.0
        bottom_right_x = center_x + width / 2.0
        bottom_right_y = center_y + height / 2.0
        inputs = tf.concat([top_left_x, top_left_y, bottom_right_x,
                            bottom_right_y, confidence, classes], axis=-1)
        boxes_dicts = self.non_max_suppression(inputs, model_size, max_output_size,
                                          max_output_size_per_class, iou_threshold, confidence_threshold)
        return boxes_dicts

    def draw_outputs(self,img, boxes, objectness, classes, nums, class_names):
        boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
        boxes = np.array(boxes)
        for i in range(nums):
            x1y1 = tuple(
                (boxes[i, 0:2] * [img.shape[1], img.shape[0]]).astype(np.int32))
            x2y2 = tuple(
                (boxes[i, 2:4] * [img.shape[1], img.shape[0]]).astype(np.int32))
            img = cv2.rectangle(img, (x1y1), (x2y2), (0, 255, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(
                classes[i])], objectness[i]), (x1y1), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1)
        return img

    def non_max_suppression(self,inputs, model_size, max_output_size,
                            max_output_size_per_class, iou_threshold,
                            confidence_threshold):
        bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
        bbox = bbox/model_size[0]
        scores = confs * class_probs
        boxes, scores, classes, valid_detections = \
            tf.image.combined_non_max_suppression(
                boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
                scores=tf.reshape(scores, (tf.shape(scores)[0], -1,
                                        tf.shape(scores)[-1])),
                max_output_size_per_class=max_output_size_per_class,
                max_total_size=max_output_size,
                iou_threshold=iou_threshold,
                score_threshold=confidence_threshold
            )
        return boxes, scores, classes, valid_detections

    def flagCreation(self, predictions, max_index, predicted_emotion, frames):
        if(predictions[0][max_index] > 0.50):

            print(self.previous)
            if(self.previous[0] == predicted_emotion):
                self.previous[1] += 1
                if(self.previous[2] == 0):
                    self.previous[2] = frames
            else:
                if(self.previous[1] > 3 and (self.previous[0] == "fear" or self.previous[0] == "happiness")):
                    self.report.append(
                        [self.previous[0], self.previous[1], self.previous[2]])

                self.previous[0] = predicted_emotion
                self.previous[1] = 1
                self.previous[2] = 0

    def videoPrediction(self, route):
        cap = cv2.VideoCapture(route)
        out_file = "new"+route
        ret, frame = cap.read()
        video_shape = (int(cap.get(3)), int(cap.get(4)))

        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, 20.0, video_shape, True)

        frames = 0
        while ret:
            predict_image = self.imagePrediction("", frame, frames)
            out.write(predict_image)
            ret, frame = cap.read()
            print(self.report)
            if(len(self.report) > 0):
                if(isinstance(self.report[-1][2], int)):
                    self.report[-1][2] = self.timeConversion(
                        (self.report[-1][2]/fps))
            frames += 1

        print(self.report)
        self.makeReport()
        print(out_file + " created")

        self.cleaningCV2()

    def timeConversion(self, time):

        time = format_timespan(time)

        print(time)
        return time

    def cleaningCV2(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def showAccuracy(self):

        for p in self.data['modelParams']:
            #print('accuracy: ' + str(p['accuracy']))
            #print('val_accuracy: ' + str(p['val_accuracy']))
            #print('loss: ' + str(p['loss']))
            #print('val_loss: ' + str(p['val_loss']))
            print('Perdia final del modelo: ' + str(p['lossEvaluate']))
            print('Accuracy final del modelo: ' + str(p['accEvaluate']))

    def trainingGraphics(self):

        fig, ax = plt.subplots(1, 2)
        train_acc = self.data['modelParams'][0]['accuracy']
        train_loss = self.data['modelParams'][0]['loss']
        fig.set_size_inches(12, 4)

        ax[0].plot(self.data['modelParams'][0]['accuracy'])
        ax[0].plot(self.data['modelParams'][0]['val_accuracy'])
        ax[0].set_title('Training Accuracy vs Validation Accuracy')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].legend(['Train', 'Validation'], loc='upper left')

        ax[1].plot(self.data['modelParams'][0]['loss'])
        ax[1].plot(self.data['modelParams'][0]['val_loss'])
        ax[1].set_title('Training Loss vs Validation Loss')
        ax[1].set_ylabel('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].legend(['Train', 'Validation'], loc='upper left')

        buf = BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"<img src='data:image/png;base64,{data}'/>"

    def makeReport(self):
        data = {}
        data['studentName'] = []
        data['studentName'].append({
            'flags': self.report
        })
        with open('report.txt', 'w') as outfile:
            json.dump(data, outfile)

        self.report = []
        self.previous = ["", 0, 0.0]
