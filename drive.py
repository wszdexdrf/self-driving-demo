# Kindly note that the libraries required are not the latest libraries. 
# The simulator wasn't updated for long hence the latest python version it supports is python 3.5.2
# and previous versions of libraries which support python 3.5.2
# This program has been written according to that



import base64
from datetime import datetime
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from openvino.inference_engine import IECore

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        
        # The current throttle of the car
        throttle = float(data["throttle"])
        
        # The current speed of the car
        speed = float(data["speed"])
        
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        
        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = np.array([image])       # the model expects 4D array

            # predict the steering angle for the image
            steering_angle, hidden = exec_net.infer(inputs={input_blob: (image, hidden)})

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':

    ie = IECore()
    
    # read the xml model description and bin file for the weights 

    net = ie.read_network(model="Driver\\Driver.xml", weights="Driver\\Driver.bin")
    exec_net = ie.load_network(network=net, device_name="CPU")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))


    # Hidden state of the model
    hidden = np.zeros((1,5))


    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
