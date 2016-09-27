import cv2.cv as cv
import cv2
import numpy as np

hue_sens=20
sat_sens=30
val_sens=15
key=None
x_co=0
y_co=0

def on_mouse(event,x,y,flag,param):
	global x_co
	global y_co
	if(event==cv.CV_EVENT_MOUSEMOVE):
		x_co=x
		y_co=y

cap = cv2.VideoCapture(1)

# cap = cv2.VideoCapture("http://192.168.1.2:8080/videofeed?dummy=file.mjpeg")
cv.NamedWindow('HSV', cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow('HS_', cv.CV_WINDOW_AUTOSIZE)

# running the classifiers
storage = cv.CreateMemStorage()

while True:

	_, frame = cap.read()
	# frame = cv2.medianBlur(frame,7)
	frame=cv2.GaussianBlur(frame,(5,5), 5)
	# frame=cv2.bilateralFilter(frame, 7, 75, 75)
	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	#hsv[:,:,2]=0
	cv2.rectangle(frame, (30,100), (150,220), (0,255,255))
	cv2.imshow('HSV', frame) 
	roi=frame[100:220, 30:150]
	roi=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	mn=cv2.mean(roi)
	# hsv[:,:,1] = cv2.threshold(hsv[:,:,1], 65, 255, cv2.THRESH_BINARY)
	#TR_MIN = np.array([0, 44, 75],np.uint8)
	#TR_MAX = np.array([27, 104, 105],np.uint8)

	TR_MIN = np.array([max(mn[0]-hue_sens,0), max(mn[1]-sat_sens,0), mn[2]-val_sens],np.uint8)
	TR_MAX = np.array([mn[0]+hue_sens, mn[1]+sat_sens, mn[2]+val_sens],np.uint8)

	#print mn[0]-hue_sens, mn[1]-sat_sens, mn[2]-val_sens,"==>",mn[0]+hue_sens, mn[1]+sat_sens, mn[2]+val_sens
	frame_threshed = cv2.inRange(hsv, TR_MIN, TR_MAX)
	# frame_threshed = cv2.adaptiveThreshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY ), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
	# frame_threshed=cv2.medianBlur(frame_threshed, 9)
	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
	#frame_threshed=cv2.erode(frame_threshed, kernel)
	#frame_threshed=cv2.dilate(frame_threshed, kernel)

	
	# frame_threshed = cv2.morphologyEx(frame_threshed,cv2.MORPH_OPEN,kernel)
	
	# cnt = cv2.findContours(frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	cv.SetMouseCallback("camera",on_mouse, 0)
	# s=cv.Get2D(cv.fromarray(hsv),y_co,x_co)
	# if s[1]<20:
	# 	hsv[:,:,1]=0
	cv2.imshow('HS_', frame_threshed) 
	
	
	# print "H:",s[0],"      S:",s[1],"       V:",s[2]
	# print mn[0],mn[1],mn[2]

	c = cv.WaitKey(1)
	if c == 27 : 
		break
	elif c==ord('h'):
		key=ord('h')
	elif c==ord('s'):
		key=ord('s')
	elif c==ord('v'):
		key=ord('v')
	elif c==ord('+'):
		if key == ord('h'):
			hue_sens+=1
		elif key==ord('s'):
			sat_sens+=1
		elif key==ord('v'):
			val_sens+=1
		print "Sensing ", hue_sens, sat_sens, val_sens
	elif c==ord('-'):
		if key == ord('h'):
			hue_sens-=1
		elif key==ord('s'):
			sat_sens-=1
		elif key==ord('v'):
			val_sens-=1
		print "Sensing ", hue_sens, sat_sens, val_sens
	