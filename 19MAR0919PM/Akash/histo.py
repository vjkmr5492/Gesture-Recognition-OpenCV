import cv2
import numpy as np
import cv2.cv as cv


"""def calculate_hue_color(values):
	global hMin,hMax
	hMin=min(values)
	hMax=max(values)
	return


def threshold_image(frame):
	global hMin,hMax
	thresh=cv2.inRange(hsv,np.array([hMin,10,10]),np.array([hMax,255,255]))
	return thresh
"""

h=np.zeros((300,256,3))
bins=np.arange(256).reshape(256,1)
color=[(255,0,0),(0,255,0),(0,0,255)]


cameraCapture=cv2.VideoCapture(1)
cv2.namedWindow('inputWindow')
cv2.namedWindow('thresholdWindow')
#cv2.setMouseCallback('inputWindow', on_Mouse)
thresh=None
success,frame=cameraCapture.read()
print 'Showing camera feed, press ESC to exit'

while success : #press ESC to end
	cv2.rectangle(frame, (30,100), (150,220), (0,255,255))
	cv2.imshow('inputWindow',frame)
	roi=frame[100:220, 30:150]
	roi=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


	#thresh=threshold_image(frame)
	#cv2.imshow('thresholdWindow',thresh)
	success,frame=cameraCapture.read()
	c = cv.WaitKey(1)
	if c == 27 : 
		break
	elif c==ord('z'):
		#hist=cv2.split(roi)
		for ch,col in enumerate(color):
			hist_item=cv2.calcHist(roi,[ch],None,[256],[0,256])
			cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
			hist=np.int32(np.around(hist_item))
			pts=np.column_stack((bins,hist))
			cv2.polylines(h,[pts],False,col)
			cv2.imshow('colorhist',h)



cv2.destroyAllWindows()
cameraCapture.release()
