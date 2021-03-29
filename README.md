# Project Overview
### Description

This project aims to take a new perspective on the problem of self driving cars. The model used in this project is fairly limited, as compared to the situtations faced by self driving cars in real scenarios, but this is just a demonstration of an idea. The car can steer through a single road (no crossroads) without traffic-lights, but those are different aspects of a car. The car model is trained and demonstrated on the lake track of the car-simulator provided [here](https://github.com/udacity/self-driving-car-sim). But the actual simulator hardly matters, what matters is the idea.

### Benefits and Features

So why make *another* self driving car model? Good question. Now, there is a plethora of self driving car model, using CNNs, etc. yet we hardly see any car without a steering wheel in the actual world. One of the many reasons for that is because the car AIs are not yet completely accurate. Even if we make the AI as good as a human, still it won't suffice. A large number of accidents happen in the real life, but in them, most of the time the person who *was* careless suffers. If we make a self driving car, and if an accident happens, the victim would be the passenger, even though it wasn't his/her fault. Therefore, a self driving car doesn't just needs to be as good as a human, it has to be almost perfect.

So my pitch here is just a simple improvement to current self driving car models. What if we use RNNs in the model? A car doesn't just needs to take decisions based on what it sees, but also what it had seen a few moments ago. So a RNN should be very useful and considerably boost the reliability of the model.

Some other points of note are:

* The car can complete entire laps of the lake track without human intervention at around the speed of 20 miles per hour.
* The car can process all the 60 fps the simulator provides in real-time. Considering that the simulator too is running on the same device, we still have headroom for more.
* This model is technically at SAE Automation level 2.

### Social Impact

I hardly have to  tell the social impacts of a fully functional self driving car. Hardly any car accidents, no one is barely late, low reaction time of machines means faster cars, and so on. It would almost be utopian.

# Documentaiton

**Most of the working of the code is explained with comments in the code itself.**

![Untitled](https://user-images.githubusercontent.com/81520912/112793770-8451c900-9083-11eb-822a-505d991f5ba1.png)

The `drive.py` consists of the main function. It receives data from the simulator, feeds it to the Model, calculates the throttle value based on the expression:
`throttle = 1 - (steering_angle)² - (speed/speed_limit)²`
The values of steering angle and throttle is passed to the simulator.

The basic structure of the Model is given below:

Convolution: 5x5, filter: 24, strides: 2x2
Convolution: 5x5, filter: 36, strides: 2x2
Convolution: 5x5, filter: 48, strides: 2x2
Convolution: 3x3, filter: 64, strides: 1x1
Convolution: 3x3, filter: 64, strides: 1x1
Flatten,
Fully connected: neurons: 100
Fully connected: neurons: 50
Simple RNN Cell: (5,) hidden state size, activation = RELU
Fully connected: neurons: 1 (output)

The fundamental idea was that the CNN layers would extract the features in the image, like the lanes, etc. and the fully connected layers would try to predict the current sitution with that. The RNN cell would then take this information, and also the Hidden state, which in some ways represent the previous situations, and try to predict the best possible steps. The final layer condenses this to one floating point number. Though note that this is just a basic idea and this may or may not represent the actual working of the model.

The dataset used to train the model was prepared by myself from the simulator. It contains images of the camera feed when a human is driving the car. There are 20 clips, each having a csv file, denoting the path of the image and its corresponding output values. These images are then loaded into a hdf5 file, which is also share, alongside the dataset, [here](https://drive.google.com/drive/folders/1wYRSwHwN4TOIaNZzX-e4kDre3rwHz_9l?usp=sharing). The losses while training the model can be seen in the jupyter notebook's outputs (attached). The final onnx file depicting the model and the compiled Intel xml and bin files are also shared in `/Driving`.

### A Journey through my troubles

The entire project was plagued by mismatching libraries. Most of the time I spent while creating this project was not in thinking and optimzing, but rather resolving dependencies. The simulator is out of date, but since it was the only open-source one which I could find, so I had to make do. The simulator, somehow, only works with python 3.5.2 and older versions of Flask and Socketio. In the newer versions, either the program wouldn't execute, or the application would just keep on waiting for information and never actually receive it. After a few painfull weeks, when I was able to connect to the simualator, I now had a different problem. Openvino 2021 doesn't support python 3.5! After downloading the older version of Openvino, and spending a couple of gruesome days setting it up, (The library wouldn't import in python, for some reason), I now realised that the compiled models wouldn't work on this older version of Intel Inference Engine! Great! Also, the Model optimizer would even read the onnx file! Fantastic! After spending few more days, creating and deleting anaconda environments, factory reseting my PC, contemplating whether I should continue or not, I was finally able to make it work, somehow. 

### Reaping the work of blood, sweat and tears

Finally, I was able to run the program and see the results. The car was able to drive, no surprises there, but the actual inference times we amazing. I tried running the model simply on the cpu using pytorch and then using Intel Inference Engine, and I could see **around 15% improvement in throughput!** It actually is the real reason why I am using RNNs. If we have to use RNNs for self driving, we would have to make a lot of inferences in a short amount of time so that the hidden state is coherent and represent some real physical state which it wouldn't if the inferences were far apart in time (even a few hundred milliseconds would be too long). With the help of OPENVINO, I could process the entire 60 frames which the simulator was sending through the model while running the simulator itself on the same machine. This goes to show how efficiently the Inference Engine runs the model. 

# Aspects of my creativity

So what makes this project different? The answer, it is an improvement. Self driving cars are not a reality on streets in cities because right now they are not *great enough* to justify their existence. So, improvements are precisely what are needed.

Because of adding RNNs, we gave the AI some way to correlate and derive conclusions from what it sees *keeping in mind the previous states* which is a very significant factor. Intel's OPENVINO provides good enough speed improvement (in inferencing), so we can make a lot of predictions and modify the hidden state each time, so that it contains something meaningful. 

Now the sky is the limit! 
