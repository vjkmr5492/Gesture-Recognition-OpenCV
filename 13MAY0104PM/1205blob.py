import cv2.cv as cv
import cv2
import numpy as np
from FingerDefectsModule import getContourFingersAndDefects

SQ_SIZE=20
hue_sens=20
sat_sens=30
val_sens=20
hue_sensy=20
sat_sensy=8
val_sensy=8
Key=None
reset=1
track=None
f_mean=np.zeros(shape=(6,3))
oldpt=None
path=[]
SQ_SIZE=20
cap = cv2.VideoCapture(0)
_, frame = cap.read()
rows=frame.shape[0]
cols=frame.shape[1]
xmin=cols-100
xmax=cols
ymin=0
ymax=100



while True:
	_, frame = cap.read()
	orig=frame
	frame=cv2.GaussianBlur(frame,(3,3),5)

	
	cv2.rectangle(orig, (int(rows/3),int(cols/2.5)), (int(rows/3+SQ_SIZE),int(cols/2.5)+SQ_SIZE), (0,255,255))
	cv2.rectangle(orig, (int(rows/4.2),int(cols/2.9)), (int(rows/4.2+SQ_SIZE),int(cols/2.9)+SQ_SIZE), (0,255,255))
	cv2.rectangle(orig, (int(rows/2.7),int(cols/3.6)), (int(rows/2.7+SQ_SIZE),int(cols/3.6)+SQ_SIZE), (0,255,255))
	cv2.rectangle(orig, (int(rows/4),int(cols/2.3)), (int(rows/4+SQ_SIZE),int(cols/2.3)+SQ_SIZE), (0,255,255))
	cv2.rectangle(orig, (int(rows/2),int(cols/2.5)), (int(rows/2+SQ_SIZE),int(cols/2.5)+SQ_SIZE), (0,255,255))
	cv2.rectangle(orig, (int(rows/2.7),int(cols/8)), (int(rows/2.7+SQ_SIZE),int(cols/8)+SQ_SIZE), (0,255,255))



	"""YUV begin
	"""
	yuv=cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
	#yuv_skin=cv2.inRange(yuv, (34,130,110), (125,153,130))

	roi_1y=yuv[int(cols/2.5):int(cols/2.5)+SQ_SIZE, int(rows/3):int(rows/3+SQ_SIZE)]
	if(reset):
		mean_1y=cv2.mean(roi_1y)
	TR_MINy = np.array([max(mean_1y[0]-hue_sens,0), max(mean_1y[1]-sat_sens,0), max(mean_1y[2]-val_sens,0)],np.uint8)
	TR_MAXy = np.array([mean_1y[0]+hue_sens, mean_1y[1]+sat_sens, mean_1y[2]+val_sens],np.uint8)
	tr_1=cv2.inRange(yuv, TR_MINy, TR_MAXy)
	
	roi_2y=yuv[int(cols/2.9):int(cols/2.9)+SQ_SIZE, int(rows/4.2):int(rows/4.2+SQ_SIZE)]
	if(reset):
		mean_2y=cv2.mean(roi_2y)
	TR_MINy = np.array([max(mean_2y[0]-hue_sens,0), max(mean_2y[1]-sat_sens,0), max(mean_2y[2]-val_sens,0)],np.uint8)
	TR_MAXy = np.array([mean_2y[0]+hue_sens, mean_2y[1]+sat_sens, mean_2y[2]+val_sens],np.uint8)
	tr_2=cv2.inRange(yuv, TR_MINy, TR_MAXy)


	roi_3y=yuv[int(cols/3.6):int(cols/3.6)+SQ_SIZE, int(rows/2.7):int(rows/2.7+SQ_SIZE)]
	if(reset):
		mean_3y=cv2.mean(roi_3y)
	TR_MINy = np.array([max(mean_3y[0]-hue_sens,0), max(mean_3y[1]-sat_sens,0), max(mean_3y[2]-val_sens,0)],np.uint8)
	TR_MAXy = np.array([mean_3y[0]+hue_sens, mean_3y[1]+sat_sens, mean_3y[2]+val_sens],np.uint8)
	tr_3=cv2.inRange(yuv, TR_MINy, TR_MAXy)


	roi_4y=yuv[int(cols/2.3):int(cols/2.3)+SQ_SIZE, int(rows/4):int(rows/4+SQ_SIZE)]
	if(reset):
		mean_4y=cv2.mean(roi_4y)
	TR_MINy= np.array([max(mean_4y[0]-hue_sens,0), max(mean_4y[1]-sat_sens,0), max(mean_4y[2]-val_sens,0)],np.uint8)
	TR_MAXy = np.array([mean_4y[0]+hue_sens, mean_4y[1]+sat_sens, mean_4y[2]+val_sens],np.uint8)
	tr_4=cv2.inRange(yuv, TR_MINy, TR_MAXy)

	roi_5y=yuv[int(cols/2.5):int(cols/2.5)+SQ_SIZE, int(rows/2):int(rows/2+SQ_SIZE)]
	if(reset):
		mean_5y=cv2.mean(roi_5y)
	TR_MINy = np.array([max(mean_5y[0]-hue_sens,0), max(mean_5y[1]-sat_sens,0), max(mean_5y[2]-val_sens,0)],np.uint8)
	TR_MAXy = np.array([mean_5y[0]+hue_sens, mean_5y[1]+sat_sens, mean_5y[2]+val_sens],np.uint8)
	tr_5=cv2.inRange(yuv, TR_MINy, TR_MAXy)

	roi_6y=yuv[int(cols/8):int(cols/8)+SQ_SIZE, int(rows/2.7):int(rows/2.7+SQ_SIZE)]
	if(reset):
		mean_6y=cv2.mean(roi_6y)
	TR_MINy = np.array([max(mean_6y[0]-hue_sens,0), max(mean_6y[1]-sat_sens,0), max(mean_6y[2]-val_sens,0)],np.uint8)
	TR_MAXy = np.array([mean_6y[0]+hue_sens, mean_6y[1]+sat_sens, mean_6y[2]+val_sens],np.uint8)
	tr_6=cv2.inRange(yuv, TR_MINy, TR_MAXy)

	tr_y = tr_1 | tr_2 | tr_3 | tr_4 | tr_5 | tr_6

	yuv_fg = tr_y

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	#yuv_fg=cv2.erode(yuv_fg,kernel)
	#yuv_fg=cv2.dilate(yuv_fg,kernel)
	yuv_fg=cv2.morphologyEx(yuv_fg,cv2.MORPH_OPEN,kernel,None,None, 2)
	yuv_fg=cv2.erode(yuv_fg,kernel)
	yuv_fg=cv2.dilate(yuv_fg,kernel)


	#cv2.imshow('yuv_fg', yuv_fg)


	""" hsv begin"""
	
	hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#hsv_skin=cv2.inRange(hsv, (0,50,0), (50,150,10))

	roi_1=hsv[int(cols/2.5):int(cols/2.5)+SQ_SIZE, int(rows/3):int(rows/3+SQ_SIZE)]
	if(reset):
		mean_1=cv2.mean(roi_1)
	TR_MIN = np.array([max(mean_1[0]-hue_sens,0), max(mean_1[1]-sat_sens,0),  max(mean_1[2]+val_sens,0)],np.uint8)
	TR_MAX = np.array([mean_1[0]+hue_sens, mean_1[1]+sat_sens, mean_1[2]+val_sens],np.uint8)
	tr_1=cv2.inRange(hsv, TR_MIN, TR_MAX)
	
	roi_2=hsv[int(cols/2.9):int(cols/2.9)+SQ_SIZE, int(rows/4.2):int(rows/4.2+SQ_SIZE)]
	if(reset):
		mean_2=cv2.mean(roi_2)
	TR_MIN = np.array([max(mean_2[0]-hue_sens,0), max(mean_2[1]-sat_sens,0),  max(mean_2[2]-val_sens,0)],np.uint8)
	TR_MAX = np.array([mean_2[0]+hue_sens, mean_2[1]+sat_sens, mean_2[2]+val_sens],np.uint8)
	tr_2=cv2.inRange(hsv, TR_MIN, TR_MAX)


	roi_3=hsv[int(cols/3.6):int(cols/3.6)+SQ_SIZE, int(rows/2.7):int(rows/2.7+SQ_SIZE)]
	if(reset):
		mean_3=cv2.mean(roi_3)
	TR_MIN = np.array([max(mean_3[0]-hue_sens,0), max(mean_3[1]-sat_sens,0),  max(mean_3[2]-val_sens,0)],np.uint8)
	TR_MAX = np.array([mean_3[0]+hue_sens, mean_3[1]+sat_sens, mean_3[2]+val_sens],np.uint8)
	tr_3=cv2.inRange(hsv, TR_MIN, TR_MAX)


	roi_4=hsv[int(cols/2.3):int(cols/2.3)+SQ_SIZE, int(rows/4):int(rows/4+SQ_SIZE)]
	if(reset):
		mean_4=cv2.mean(roi_4)
	TR_MIN = np.array([max(mean_4[0]-hue_sens,0), max(mean_4[1]-sat_sens,0), max(mean_4[2]-val_sens,0)],np.uint8)
	TR_MAX = np.array([mean_4[0]+hue_sens, mean_4[1]+sat_sens,  mean_4[2]+val_sens],np.uint8)
	tr_4=cv2.inRange(hsv, TR_MIN, TR_MAX)

	roi_5=hsv[int(cols/2.5):int(cols/2.5)+SQ_SIZE, int(rows/2):int(rows/2+SQ_SIZE)]
	if(reset):
		mean_5=cv2.mean(roi_5)
	TR_MIN = np.array([max(mean_5[0]-hue_sens,0), max(mean_5[1]-sat_sens,0),  max(mean_5[2]-val_sens,0)],np.uint8)
	TR_MAX = np.array([mean_5[0]+hue_sens, mean_5[1]+sat_sens,  mean_5[2]+val_sens],np.uint8)
	tr_5=cv2.inRange(hsv, TR_MIN, TR_MAX)

	roi_6=hsv[int(cols/8):int(cols/8)+SQ_SIZE, int(rows/2.7):int(rows/2.7+SQ_SIZE)]
	if(reset):
		mean_6=cv2.mean(roi_6)
	TR_MIN = np.array([max(mean_6[0]-hue_sens,0), max(mean_6[1]-sat_sens,0),  max(mean_6[2]-val_sens,0)],np.uint8)
	TR_MAX = np.array([mean_6[0]+hue_sens, mean_6[1]+sat_sens,  mean_6[2]+val_sens],np.uint8)
	tr_6=cv2.inRange(hsv, TR_MIN, TR_MAX)
	
	tr = tr_1|tr_2|tr_3|tr_4|tr_5|tr_6

	hsv_fg = tr
	#yuv_fg=cv2.erode(yuv_fg,kernel)
	#yuv_fg=cv2.dilate(yuv_fg,kernel)
	hsv_fg=cv2.morphologyEx(hsv_fg,cv2.MORPH_CLOSE,kernel,None,None, 2)
	#hsv_fg=cv2.morphologyEx(hsv_fg,cv2.MORPH_OPEN,kernel,None,None, 1)
	hsv_fg=cv2.dilate(hsv_fg,kernel)
	#cv2.imshow('hsv_fg',hsv_fg)


	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	#skin = cv2.dilate(skin,kernel)
	#skin = cv2.erode(skin,kernel)
	#skin=cv2.medianBlur(skin, 3)

	hand=yuv_fg|hsv_fg
	hand2=yuv_fg&hsv_fg
	#cv2.imshow('hand',hand)
	#cv2.imshow('hand2',hand2)

	
	
	cfd,nfingers,fingers=getContourFingersAndDefects(frame,hand)
	#cv2.circle(cfd,(trackpt[0],trackpt[1]),5,(0,0,255),-1) 
	cv2.rectangle(cfd,(xmin,ymin),(xmax,ymax),(0,255,0),2) 
	cv2.imshow('ContourFingersAndDefects',cfd)


	
	if track ==None:
		track=np.ones((rows,cols,3), np.uint8)

	trackpt=[0,0]
	for x in fingers:
		if x[0] > xmin and x[0] <xmax:
			if x[1] > ymin and x[1] <ymax:
				trackpt=x
			
	
	if trackpt is not None:
		if oldpt is None:
			xmin,ymin=trackpt[0]-50,trackpt[1]-50
			xmax,ymax=trackpt[0]+50,trackpt[1]+50
			cv2.rectangle(cfd,(xmin,ymin),(xmax,ymax),(0,255,0),2)
			midx=int((xmax+xmin)/2)
			midy=int((ymax+ymin)/2)
			cv2.circle(track,(midx,midy),5,(0,0,255),-1) 
			oldpt=(midx,midy)
			path.append([midx,midy])
		else:
			xmin,ymin=trackpt[0]-50,trackpt[1]-50
			xmax,ymax=trackpt[0]+50,trackpt[1]+50
			cv2.rectangle(cfd,(xmin,ymin),(xmax,ymax),(0,255,0),2)
			midx=int((xmax+xmin)/2)
			midy=int((ymax+ymin)/2)
			print midy,midx
			cv2.line(track, oldpt , (midx,midy) , (0,0,255), 8)
			oldpt=(midx,midy)
			path.append([midx,midy])
	else:
		"""
		if oldpt is not None:
			print "continue the gesture in 5 seconds"
			for i in range(5):
				print i+"..."
				for j in range(50):
					pass
		"""
		#save gesture
		track=np.ones((rows,cols,3), np.uint8)
		path=[]



	cv2.imshow('path',track)
	
	

	
	if(reset):
		reset=0
	cv2.imshow('YUV', orig) 
	c = cv.WaitKey(1)
	if c == 27 : 
		break
	elif c==ord('z'):
		reset=1
	




cv2.destroyAllWindows()
cap.release()