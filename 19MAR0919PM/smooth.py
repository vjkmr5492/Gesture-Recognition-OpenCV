import cv2
import numpy as np
import cv2.cv as cv


def calculate_roi_color(hsv):
	roi=hsv[110:200,10:100]
	#roi[:,:,2]=0
	avg,std=cv2.meanStdDev(roi)
	#avg,std=cv2.meanStdDev(roi)
	std=std*0.8
	maxColor=[round(sum(x)) for x in zip(avg,std)]

	minColor=[round(u-v) for (u,v) in zip(avg,std)]
	maxColor[2]=0
	minColor[2]=0
	print maxColor, "TO ", minColor
	return [maxColor,minColor]


def threshold_image(hsv):
	maxColor,minColor=calculate_roi_color(hsv)
	hsv[:,:,2]=0
	thresh=cv2.inRange(hsv,np.array(minColor),np.array(maxColor))
	return thresh






values=[]
hMax=0
hMin=0

cameraCapture=cv2.VideoCapture(0)
cv2.namedWindow('inputWindow')
cv2.namedWindow('thresholdWindow')
#cv2.setMouseCallback('inputWindow', on_Mouse)
thresh=None
success,frame=cameraCapture.read()
print 'Showing camera feed, press ESC to exit'

while success and cv2.waitKey(1)!=27: #press ESC to end
	cv2.imshow("inputWindow", frame)
	cv2.rectangle(frame,(10,100),(110,200),(0,0,255))
	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	

	
	
	thresh=threshold_image(hsv)

	
	#cv2.imshow('inputWindow',frame)
	#thresh=threshold_image(frame)
	cv2.imshow('thresholdWindow',thresh)
	success,frame=cameraCapture.read()

cv2.destroyWindow('inputWindow')
cv2.destroyWindow('thresholdWindow')


