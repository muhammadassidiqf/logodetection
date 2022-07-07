import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import os
import sys
import time
import cv2

class Video:
    def __init__(self, filepath):
        self.filepath = filepath
    def get_frame(self, filepath):
        cap = cv2.VideoCapture(filepath)
        model = load_model('static/models/model.hdf5')
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        while True:

            r, frame = cap.read() 
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
            
            ret,jpg=cv2.imencode('.jpg',frame)
            yield cv2.imencode('.jpg', jpg)[1].tobytes()