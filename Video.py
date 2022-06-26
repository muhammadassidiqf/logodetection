import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import os
import sys
import time
import cv2

class Video():
    def __init__(self, filepath ):
        self.filepath = filepath
        self.frames = []
    def get_frame(self):
        cap = cv2.VideoCapture(self.filepath)
        print(self.filepath)
        model = load_model('static/models/model.hdf5')
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        r, frame = cap.read() 
        while True:
            img = cv2.resize(frame, (224,224))

            img_array = image.img_to_array(img)
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            frame_model = img_array_expanded_dims
            classes = model.predict(frame_model)
            res = ""
            if classes[0][0] > 0.8:
                result = classes[0][0] * 100
                res += "Detected - " + str(result)
            else:
                result = classes[0][1] * 100
                res += "Not Detected - " + str(result)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, res, (25, 30), font, 1, (255, 255, 0), 1, cv2.LINE_AA)
            # print(res)
            ret,jpg=cv2.imencode('.jpg', frame)
            yield jpg.tobytes()
    def save_video(self, filename):
        cap = cv2.VideoCapture(filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        out = cv2.VideoWriter('static/results/'+filename, fourcc, 20, size)
        while True:
            ret, frame = cap.read()
            if ret:
                img = cv2.resize(frame, (224,224))
                img_array = image.img_to_array(img)
                img_array_expanded_dims = np.expand_dims(img_array, axis=0)
                frame_model = img_array_expanded_dims
                model = load_model('static/models/model.hdf5')
                model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])
                classes = model.predict(frame_model)
                res = ""
                if classes[0][0] > 0.8:
                    result = classes[0][0] * 100
                    res += "Detected - " + str(result)
                else:
                    result = classes[0][1] * 100
                    res += "Not Detected - " + str(result)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, res, (25, 30), font, 1, (255, 255, 0), 1, cv2.LINE_AA)
                # ret, jpeg = cv2.imencode('.jpg', frame)
                
                out.write(frame)
                cv2.imshow('preview', frame)
            # framet = jpeg.tobytes()
            else:
                out.release() 
                return True                       
    def modelnya(self, framemodel):
        model = load_model('static/models/model.hdf5')
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        classes = model.predict(framemodel)
        res = ""
        if classes[0][0] > 0.8:
            result = classes[0][0] * 100
            res += "Detected - " + str(result)
        else:
            result = classes[0][1] * 100
            res += "Not Detected - " + str(result)
        return res