import cv2.cv as cv
import cv2
import numpy as np
# ip-cam
import base64
import time
import urllib2

#ip-class
class ipCamera(object):

    def __init__(self, url, user=None, password=None):
        self.url = url
        auth_encoded = base64.encodestring('%s:%s' % (user, password))[:-1]

        self.req = urllib2.Request(self.url)
        self.req.add_header('Authorization', 'Basic %s' % auth_encoded)

    def get_frame(self):
        response = urllib2.urlopen(self.req)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        return frame


class Camera(object):

    def __init__(self, camera=0):
        self.cam = cv2.VideoCapture(camera)
        if not self.cam:
            raise Exception("Camera not accessible")

        self.shape = self.get_frame().shape

    def get_frame(self):
        _, frame = self.cam.read()
        return frame
#/ip-class


x_co=0
y_co=0

def on_mouse(event,x,y,flag,param):
	global x_co
	global y_co
	if(event==cv.CV_EVENT_MOUSEMOVE):
		x_co=x
		y_co=y

cap = cv2.VideoCapture("http://192.168.1.2:8080/videofeed?dummy=file.mjpeg")

cv.NamedWindow('HSV', cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow('HS_', cv.CV_WINDOW_AUTOSIZE)

# running the classifiers
storage = cv.CreateMemStorage()

while True:

	_, frame = cap.read()
	#frame = cv2.medianBlur(frame,5)
	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	hsv[:,:,2]=0
	cv2.imshow('HSV', hsv) 
	# hsv[:,:,1] = cv2.threshold(hsv[:,:,1], 65, 255, cv2.THRESH_BINARY)
	# TR_MIN = np.array([5, 50, 50],np.uint8)
	# TR_MAX = np.array([15, 255, 255],np.uint8)
	# frame_threshed = cv2.inRange(hsv, TR_MIN, TR_MAX)
	cv.SetMouseCallback("camera",on_mouse, 0)
	s=cv.Get2D(cv.fromarray(hsv),y_co,x_co)
	if s[1]<20:
		hsv[:,:,1]=0
	cv2.imshow('HS_', hsv) 
	
	
	print "H:",s[0],"      S:",s[1],"       V:",s[2]
	

	c = cv.WaitKey(1)
	if c == 27 : 
		break
	