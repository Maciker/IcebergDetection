import numpy as np
import imutils
import cv2
import os
from keras.models import load_model
from flask import Flask, render_template, url_for, Response
import tensorflow as tf
global graph
global writer
graph = tf.get_default_graph()
writer = None

model = load_model('iceberg.h5')


from skimage.transform import resize

vals = ['Ship', 'Iceberg']

app = Flask(__name__)

print("[INFO] accessing video stream...")
vs = cv2.VideoCapture("iceberg1.mp4")

pred=""
def detect(frame):
        img = resize(frame,(75,75))
        img = np.expand_dims(img,axis=0)
        if(np.max(img)>1):
            img = img/255.0
        with graph.as_default():
                prediction = model.predict_classes(img)
        pred=vals[prediction[0][0]]
        if pred:
                text = "Beware!! Iceberg ahead."
        else:
                text = "You are safe! It's a Ship."
        return text


# initialize the video stream and pointer to output video file


@app.route('/')
def index():
    return render_template('index.html')

def gen():
        while True:
            # read the next frame from the file
            (grabbed, frame) = vs.read()
            
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break
            # resize the frame and then de
            #print(ix)
            #for x in vals:
            data = detect(frame)
            
            # output frame
            text = data
            cv2.putText(frame, text, (10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
            cv2.imwrite("1.jpg",frame)

            # check to see if the output frame should be displayed to our
            # screen
            # show the output frame
            #cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(r"output.avi", fourcc, 25,(frame.shape[1], frame.shape[0]), True)
            # if an output video file path has been supplied and the video
            # writer has not been initialized, do so now
            #if writer is None:
                # initialize our video writer
                
            # if the video writer is not None, write the frame to the output
            # video file
            #if writer is not None:
            #    writer.write(frame)
            if(pred==1):
                playsound(r'cut-alert.mp3')
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                                bytearray(encodedImage) + b'\r\n')
        #cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
