import cv2.cv as cv
import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv.NamedWindow('YUV', cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow('SKIN', cv.CV_WINDOW_AUTOSIZE)

cv2.createTrackbar('Y_min','YUV',0,255,nothing)
cv2.createTrackbar('Y_max','YUV',0,255,nothing)
cv2.createTrackbar('Cb_min','YUV',0,255,nothing)
cv2.createTrackbar('Cb_max','YUV',0,255,nothing)
cv2.createTrackbar('Cr_min','YUV',0,255,nothing)
cv2.createTrackbar('Cr_max','YUV',0,255,nothing)

while True:
	_, frame = cap.read()
	
	frame=cv2.GaussianBlur(frame,(3,3), 5)
	y_min=cv2.getTrackbarPos('Y_min','YUV')
	Cb_min=cv2.getTrackbarPos('Cb_min','YUV')
	Cr_min=cv2.getTrackbarPos('Cr_min','YUV')

	y_max=cv2.getTrackbarPos('Y_max','YUV')
	Cb_max=cv2.getTrackbarPos('Cb_max','YUV')
	Cr_max=cv2.getTrackbarPos('Cr_max','YUV')

	yuv=cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
	# skin=cv2.inRange(yuv, (0,133,77), (255,173,127))
	skin=cv2.inRange(yuv, (y_min,Cb_min,Cr_min), (y_max,Cb_max,Cr_max))
	skin=cv2.medianBlur(skin, 3)
	cv2.imshow('YUV', frame) 
	cv2.imshow('SKIN',   skin  )
	c = cv.WaitKey(1)
	if c == 27 : 
		break

