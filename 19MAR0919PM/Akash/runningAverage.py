import cv2.cv as cv
import cv2
import numpy as np





c = cv2.VideoCapture(1)
_,frame=c.read()

avg=np.float32(frame)

while(1):
	_,frame=c.read()
	"""
	0.001999"""
	cv2.accumulateWeighted(frame,avg,0.01)
	res=cv2.convertScaleAbs(avg)

	diff=cv2.absdiff(frame,res)
	diff=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
	_,thresh=cv2.threshold(diff,50,255,cv2.THRESH_BINARY)
	# thresh=diff
	
	cv2.imshow('img',frame)
	cv2.imshow('threshold',thresh)
	cv2.imshow('avg',res)
	if(cv2.waitKey(1) == 27):
		break

cv2.destroyAllWindows()
c.release()