import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import os
import sys
import time
import cv2

def prepare_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    # img_norm = tf.keras.utils.img_to_array(img).astype(np.float32)/255
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims

model = load_model('static/models/model.hdf5')
model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,output_names, freeze_var_names)
        return frozen_graph

# frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in model.outputs])
# tf.compat.v1.train.write_graph(frozen_graph, "models", "tf_model.pb", as_text=False)
filepath = 'bca.png'


# #matriks citra asli
# inp = tf.keras.utils.load_img(filepath)
# img_array = tf.keras.utils.img_to_array(inp)

# filepath = 'ilustrasilaga.jpg'
# start = time.time()

# # #matriks pre-processing
# img = prepare_image(filepath)

# #proses prediksi
# classes = model.predict(img)

# timing = time.time() - start

# print("processing time: ", timing)

# print(classes[0][0])
# res = []   
# if classes[0][0] > 0.8:
#     result = classes[0][0] * 100
#     res.append(("Detected",result))
#     # print("Daun Padi Sehat")
# else:
#     result = classes[0][1] * 100
#     res.append(("Not Detected",result))
# print(res)

cap = cv2.VideoCapture('videoplayback.mp4')

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
    cv2.imshow('preview', frame)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2

# img = cv2.imread('static/results/TESTING/bri/data_bri_resized_height_42.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

# contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# x1, y1, w, h = cv2.boundingRect(contours[0])
# x2, y2 = x1 + w, y1 + h
# print((x1, y1), (x2, y2)) 
# cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# cv2.imwrite('res.jpg', img)