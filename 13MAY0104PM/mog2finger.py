import numpy as np
import cv2, os, time
from FingerDefectsModule import getContourFingersAndDefects
from HMMDetector import detect, gen_cb, save_object, directory, train

gname="UL"



cap = cv2.VideoCapture(1)

fgbg = cv2.BackgroundSubtractorMOG2(0, 256, 0)
Key=None
track=None
oldpt=None
path=[]
i=0
while(1):
	ret, frame = cap.read()
	rows=frame.shape[0]
	cols=frame.shape[1]

	frame=cv2.GaussianBlur(frame,(5,5), 10)
	fgmask = fgbg.apply(frame )

	cv2.imshow('frame',fgmask)
	if i<20:
		i+=1
		continue
	
	cfd,nFingers,trackpt=getContourFingersAndDefects(frame,fgmask)
	cv2.circle(cfd,(trackpt[0],trackpt[1]),5,(0,0,255),-1) 
	cv2.imshow('ContourFingersAndDefects',cfd)

	# print "NFINGERS", nFingers
	if track ==None:
		track=np.ones((rows,cols,3), np.uint8)

	if nFingers > 0:

		if oldpt is None:
			cv2.circle(track,(trackpt[0],trackpt[1]),5,(0,0,255),-1) 
			oldpt=(trackpt[0],trackpt[1])
			path.append(trackpt)
		else:
			cv2.line(track, oldpt , (trackpt[0],trackpt[1]) , (0,0,255), 8)
			oldpt=(trackpt[0],trackpt[1])
			path.append(trackpt)
	else:
		track=np.ones((rows,cols,3), np.uint8)
		# print len(path)
		if(len(path)>20):
			detect(path)
		path=[]
		oldpt=None

	cv2.imshow('path',track)
	

	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break
	elif k==ord('s'):
		directory(gname)
		fname=int(time.time())
		save_object(path, "./"+gname+"/"+str(fname)+".raw")
		# print path
	elif k==ord('g'):
		gen_cb()
	elif k==ord('t'):
		train()
	elif k==ord('v'):
		detect(path)

cap.release()
cv2.destroyAllWindows()