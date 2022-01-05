# USTrafficSignRecognitionBox


## Installation
### Raspberry PI
* sudo apt-get install python3-opencv
### Install Darknet
* git clone https://github.com/AlexeyAB/darknet
* cd darknet
* make
* cd ..
### Install USTrafficSignRecognitionBox
* git clone https://github.com/kein0r/USTrafficSignRecognitionBox.git 
* cd USTrafficSignRecognitionBox
* wget https://pjreddie.com/media/files/yolov3-tiny.weights
* virtualenv venv
* source venv/bin/activate
* pip install -r requirements.txt
