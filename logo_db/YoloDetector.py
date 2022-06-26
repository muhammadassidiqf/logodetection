import numpy as np
import time
import cv2
import imutils
from imutils.video import FPS
from imutils.video import VideoStream

# from main import LogoDB
from logo_db.main import LogoDB


class YoloDetector():

    # def __init__(self, id_model, video_id, thresh):
    #     id_model = id_model
    #     video_id = video_id
    #     thresh = thresh

    # @classmethod
    
    def main(self, id_model, video_id, thresh, ads):
        data_model = LogoDB.get_one_model(self=LogoDB, model_id=id_model)
        modelpath = './static/models/'

        data_video = LogoDB.get_one_video(self=LogoDB, video_id=video_id)
        # print(data_video)
        INPUT_FILE='./static/input/' + data_video['video_filename_awal']
        OUTPUT_FILE='./static/output/' + data_video['video_filename_akhir']
        LABELS_FILE= modelpath + data_model['model_nama'] + '/'+ data_model['model_label']
        CONFIG_FILE= modelpath +  data_model['model_nama'] + '/'+ data_model['model_cfg']
        WEIGHTS_FILE= modelpath +  data_model['model_nama'] + '/'+ data_model['model_weights']
        CONFIDENCE_THRESHOLD=thresh

        H=None
        W=None
        arr_dur = []
        fps = FPS().start()

        LABELS = open(LABELS_FILE).read().strip().split("\n")

        np.random.seed(4)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
            dtype="uint8")


        net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
        # print(net)
        vs = cv2.VideoCapture(INPUT_FILE)
        fpse = vs.get(cv2.CAP_PROP_FPS)
        frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fpse

        print('fps = ' + str(fpse))
        print('number of frames = ' + str(frame_count))
        print('duration (S) = ' + str(duration))
        minutes = int(duration/60)
        seconds = duration%60
        print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, fpse,
            (800, 600), True)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        cnt =0;
        while True:
            startTime = time.time()
            
            cnt+=1
            # print ("Frame number", cnt)
            (grabbed, image) = vs.read()
            if grabbed:
            
                blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                    swapRB=True, crop=False)
                net.setInput(blob)
                if W is None or H is None:
                    (H, W) = image.shape[:2]
                layerOutputs = net.forward(ln)
                # print(layerOutputs)
                # initialize our lists of detected bounding boxes, confidences, and
                # class IDs, respectively
                boxes = []
                confidences = []
                classIDs = []

                # loop over each of the layer outputs
                for output in layerOutputs:
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability) of
                        # the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > CONFIDENCE_THRESHOLD:
                            # scale the bounding box coordinates back relative to the
                            # size of the image, keeping in mind that YOLO actually
                            # returns the center (x, y)-coordinates of the bounding
                            # box followed by the boxes' width and height
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top and
                            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates, confidences,
                            # and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)

                idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
                    CONFIDENCE_THRESHOLD)
                # print(idxs)
                if len(idxs) > 0:
                    for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        color = [int(c) for c in COLORS[classIDs[i]]]

                        cv2.rectangle(image, (x, y), (x + w, y + h), (103, 247, 20), 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (103, 247, 20), 2)

                    dur = cnt/fpse
                    
                    arr_dur.append(dur)

                    # start

                    # print('array durasi : ', arr_dur)
                else:
                    
                    if arr_dur:
                        
                        start = float(arr_dur[0])
                        s_min = int(start/60)
                        s_sec  = round(start%60, 1)
                        #end
                        end = float(arr_dur[-1])
                        e_min = int(end/60)
                        e_sec  = round(end%60, 1)

                        #durasi
                        durasinya = round(end - start,1)
                        d_min = int(durasinya/60)
                        d_sec  = round(durasinya%60, 1)

                        #ads per menit nya
                        adspermenit = round((int(ads)/60) * durasinya, 1)
                        
                        arr_dur_for_add = [ data_video['video_id'], str(s_min) +" : " + str(s_sec),  str(e_min) +" : " + str(e_sec), str(d_min) +" : " + str(d_sec), durasinya, adspermenit]

                        LogoDB.add_durasi(self=LogoDB, arr_durasi=arr_dur_for_add)

                        print('detect at (M:S) [ start = ' + str(s_min) +" : " + str(s_sec) + ' || end = '  + str(e_min) +" : " + str(e_sec) +" || duration = " + str(d_min) +" : " + str(d_sec) +" ]")
                    arr_dur = []
                
                endTime = time.time()
                if cnt == 1:
                    elapsetime = (endTime - startTime) * frame_count
                    print('Estimasi proses :', elapsetime)
                    LogoDB.add_estimated_time(self=LogoDB, data=[elapsetime, video_id])
                for_txt_duration = cnt/fpse
                txt_min = int(for_txt_duration/60)
                txt_sec  = round(for_txt_duration%60, 3)

                txt_duration = "durasi (M:S) [ " + str(txt_min) + " : " + str(txt_sec) + " ]"
                color2 = (0, 0, 255)
                
                cv2.putText(image, txt_duration, (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2, 2)
                # cv2.imshow("output", cv2.resize(image,(800, 600)))
                writer.write(cv2.resize(image,(800, 600)))
                fps.update()
                # cv2.destroyAllWindows()

                # key = cv2.waitKey(1) & 0xFF
                # if key == ord("q"):
                #     break
            else:
                fps.stop()
                writer.release()
                vs.release()
                return fpse

            
        # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # # do a bit of cleanup
        # cv2.destroyAllWindows()

        # # release the file pointers
        # print("[INFO] cleaning up...")
        # writer.release()
        # vs.release()
        # return True
    
if __name__ == "__main__":

    yolo = YoloDetector()
    yolo.main(id_model=1, video_id='persib-vs-persija_Trim_1656057014.1840572.mp4', thresh=0.5, ads=100000)
    # print(getmodel['model_nama'])
