import cv2
from utils import detector_utils as detector_utils
from utils import object_id_utils as id_utils
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
        self.loadJsonModels()
        self.initCV2()
        self.emotions = ['anger', 'disgust', 'fear',
                         'happiness', 'neutral', 'sadness', 'surprise']
        self.names = ['mobile phone', 'person']
        with open('models/emotions/modelParams.txt') as json_file:
            self.data = json.load(json_file)
        self.reportEmotions = []
        self.reportObjects = []
        self.reportFaces = []
        self.reportBehavior = []
        self.previousEmotion = ["", 0, 0.0]
        self.previousObjects = [[],[],[]]
        self.classNames = self.loadNames()
        self.encodeListKnown = self.encode()
        self.frames = 0
        self.objectFrames = 0
        self.studentName = ""
        self.frame_processed = 0
        self.score_thresh = 0.7
        self.num_hands_detect = 10
        self.num_classes = 1
        self.label_path = "hand_inference_graph/hand_label_map.pbtxt"
        self.frozen_graph_path = "hand_inference_graph/frozen_inference_graph.pb"
        self.object_refresh_timeout = 3
        self.seen_object_list = {}

    def loadJsonModels(self):
        #Emotion model
        json_file = open('models/emotions/fer.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.emotionsModel = model_from_json(loaded_model_json)
        self.emotionsModel.load_weights('models/emotions/fer.h5')
        #Object detection model
        json_file = open('models/objects/examanager.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.objectModel = model_from_json(loaded_model_json)
        self.objectModel.load_weights('models/objects/yolov3-examanager_weights.tf')

    def initCV2(self):
        self.face_haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.cap = cv2.VideoCapture(0)

    def emotionDetection(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05,
                                                                 minNeighbors=5, minSize=(30, 30),
                                                                 flags=cv2.CASCADE_SCALE_IMAGE)
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
            self.flagCreationEmotions(predictions, max_index,
                                      predicted_emotion, self.frames)
        if(len(self.reportEmotions) > 0):
            if(isinstance(self.reportEmotions[-1][2], int)):
                self.reportEmotions[-1][2] = self.timeConversion(
                    (self.reportEmotions[-1][2]/3))
        return img

    def objectDetection(self, img):
        img_pixels = image.img_to_array(img)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        objectPrediction = self.objectModel.predict(
            tf.image.resize(img_pixels, [416, 416]))

        boxes, scores, classes, nums = self.output_boxes(
            objectPrediction, (413, 413, 3),
            max_output_size=40,
            max_output_size_per_class=20,
            iou_threshold=0.5,
            confidence_threshold=0.3)
        imagobj = np.squeeze(img_pixels)
        img = self.draw_outputs(imagobj, boxes, scores,
                                classes, nums, self.names)
        return img

    def facialDetection(self, img):
        #imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces_detected = self.face_haar_cascade.detectMultiScale(imgS, scaleFactor=1.05,
                                                                 minNeighbors=5, minSize=(30, 30),
                                                                 flags=cv2.CASCADE_SCALE_IMAGE)
        #facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, faces_detected)

        for encodeFace, faceLoc in zip(encodesCurFrame, faces_detected):
            matches = face_recognition.compare_faces(
                self.encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(
                self.encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if faceDis[matchIndex] < 0.50:

                name = self.classNames[matchIndex].upper()
                name = name.split('_')[0]
                self.studentName = name

            else:
                name = 'Unknown'
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        return img

    def behaviorDetection(self,frame):
        
      
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detection_graph, sess, category_index = detector_utils.load_inference_graph(self.num_classes, self.frozen_graph_path, self.label_path)
        sess = tf.compat.v1.Session(graph=detection_graph)
        boxes, scores, classes = detector_utils.detect_objects(frame, detection_graph, sess)
            
        tags = detector_utils.get_tags(classes, category_index, self.num_hands_detect, self.score_thresh, scores, boxes, frame)
            
        if (len(tags) > 0):
            id_utils.get_id(tags, self.seen_object_list)
  
        id_utils.refresh_seen_object_list(self.seen_object_list, self.object_refresh_timeout)
        detector_utils.draw_box_on_image_id(tags, frame) 
            
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


        return frame
        #return img

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

    def output_boxes(self, inputs, model_size, max_output_size, max_output_size_per_class,
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

    def draw_outputs(self, img, boxes, objectness, classes, nums, class_names):
        boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
        boxes = np.array(boxes)
        detections = []
        for i in range(nums):
            x1y1 = tuple(
                (boxes[i, 0:2] * [img.shape[1], img.shape[0]]).astype(np.int32))
            x2y2 = tuple(
                (boxes[i, 2:4] * [img.shape[1], img.shape[0]]).astype(np.int32))
            detectedName = class_names[int(classes[i])]
            detections.append(detectedName)
            img = cv2.rectangle(img, (x1y1), (x2y2), (0, 255, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(detectedName, objectness[i]), (x1y1), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1)
        self.flagCreationObjects(detections)
        return img

    def non_max_suppression(self, inputs, model_size, max_output_size,
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

    def flagCreationEmotions(self, predictions, max_index, predicted_emotion, frames):
        if(predictions[0][max_index] > 0.50):

            if(self.previousEmotion[0] == predicted_emotion):
                self.previousEmotion[1] += 1
                if(self.previousEmotion[2] == 0):
                    self.previousEmotion[2] = frames
            else:
                if(self.previousEmotion[1] > 3 and (self.previousEmotion[0] == "fear" or self.previousEmotion[0] == "happiness")):
                    self.reportEmotions.append(
                        [self.previousEmotion[0], self.previousEmotion[1], self.previousEmotion[2]])

                self.previousEmotion[0] = predicted_emotion
                self.previousEmotion[1] = 1
                self.previousEmotion[2] = 0

    def flagCreationObjects(self, detections):
        for detection in detections:
            if detection == "mobile phone":
                if detection in self.previousObjects[0]:
                    if detection in self.previousObjects[1]:
                        if detection in self.previousObjects[2]:
                            self.reportObjects.append(["Mobile phone in frame", self.timeConversion(self.frames/3)])
            if detections.count("person") >= 2:
                if self.previousObjects[0].count("person") >= 2:
                    if self.previousObjects[1].count("person") >= 2:
                        if self.previousObjects[2].count("person") >= 2:
                            self.reportObjects.append(["Multiple people in frame", self.timeConversion(self.frames/3)])

        self.previousObjects[2] = self.previousObjects[1]
        self.previousObjects[1] = self.previousObjects[0]
        self.previousObjects[0] = detections
        

    def timeConversion(self, time):

        time = format_timespan(time)

        return time

    def cleaningCV2(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def makeReport(self):
        data = {}
        data[self.studentName] = []
        data[self.studentName].append({
            'flagsEmotions': self.reportEmotions,
            'flagsObjects': self.reportObjects,
            'flagsFaces': self.reportFaces,
            'flagsBehavior': self.reportBehavior
        })

        with open(f'report_{self.studentName}.txt', 'w') as outfile:
            json.dump(data, outfile)

        self.reportEmotions = []
        self.previousEmotion = ["", 0, 0.0]
