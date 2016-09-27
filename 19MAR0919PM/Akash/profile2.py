import cv2.cv as cv
import cv2
import numpy as np

SQ_SIZE=25
hue_sens=20
sat_sens=30
val_sens=15
Key=None
reset=1
f_mean=np.zeros(shape=(6,3))
cap = cv2.VideoCapture(0)
cv.NamedWindow('HSV', cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow('HS_', cv.CV_WINDOW_AUTOSIZE)

while True:
	_, frame = cap.read()
	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	color=cv2.cvtColor(frame,cv2.COLOR_HSV2BGR)
	frame=hsv
	orig=frame
	frame=cv2.GaussianBlur(frame,(5,5), 5)
	rows=frame.shape[0]
	cols=frame.shape[1]
	cv2.rectangle(frame, (int(rows/3),int(cols/2.5)), (int(rows/3+SQ_SIZE),int(cols/2.5)+SQ_SIZE), (0,255,255))
	cv2.rectangle(frame, (int(rows/4.2),int(cols/2.9)), (int(rows/4.2+SQ_SIZE),int(cols/2.9)+SQ_SIZE), (0,255,255))
	cv2.rectangle(frame, (int(rows/2.7),int(cols/3.6)), (int(rows/2.7+SQ_SIZE),int(cols/3.6)+SQ_SIZE), (0,255,255))
	cv2.rectangle(frame, (int(rows/4),int(cols/2.3)), (int(rows/4+SQ_SIZE),int(cols/2.3)+SQ_SIZE), (0,255,255))
	cv2.rectangle(frame, (int(rows/2),int(cols/2.5)), (int(rows/2+SQ_SIZE),int(cols/2.5)+SQ_SIZE), (0,255,255))
	cv2.rectangle(frame, (int(rows/2.7),int(cols/8)), (int(rows/2.7+SQ_SIZE),int(cols/8)+SQ_SIZE), (0,255,255))

	roi_1=orig[int(cols/2.5):int(cols/2.5)+SQ_SIZE, int(rows/3):int(rows/3+SQ_SIZE)]
	rh1 = cv2.calcHist([roi_1],[1], None, [256], [0, 256] )
	cv2.normalize(rh1,rh1,0,255,cv2.NORM_MINMAX)
	if(reset):
		mean_1=cv2.mean(roi_1)
	# TR_MIN = np.array([0, max(mean_1[1]-sat_sens,0), mean_1[2]-val_sens],np.uint8)
	# TR_MAX = np.array([mean_1[0]+hue_sens, mean_1[1]+sat_sens, mean_1[2]+val_sens],np.uint8)
	# tr_1=cv2.inRange(orig, TR_MIN, TR_MAX)
	tr_1=cv2.calcBackProject([frame],[1],rh1,[0,256],1)
	cv2.imshow('hist1',tr_1)
	
	roi_2=orig[int(cols/2.9):int(cols/2.9)+SQ_SIZE, int(rows/4.2):int(rows/4.2+SQ_SIZE)]
	rh2 = cv2.calcHist([roi_2],[0], None, [180], [0, 180] )
	cv2.normalize(rh2,rh2,0,255,cv2.NORM_MINMAX)
	if(reset):
		mean_2=cv2.mean(roi_2)
	# TR_MIN = np.array([0, max(mean_2[1]-sat_sens,0), mean_2[2]-val_sens],np.uint8)
	# TR_MAX = np.array([mean_2[0]+hue_sens, mean_2[1]+sat_sens, mean_2[2]+val_sens],np.uint8)
	# tr_2=cv2.inRange(orig, TR_MIN, TR_MAX)
	tr_2=cv2.calcBackProject([frame],[0],rh2,[0,180],1)



	roi_3=orig[int(cols/3.6):int(cols/3.6)+SQ_SIZE, int(rows/2.7):int(rows/2.7+SQ_SIZE)]
	rh3 = cv2.calcHist([roi_3],[0], None, [180], [0, 180] )
	cv2.normalize(rh3,rh3,0,255,cv2.NORM_MINMAX)
	if(reset):
		mean_3=cv2.mean(roi_3)
	# TR_MIN = np.array([0, max(mean_3[1]-sat_sens,0), mean_3[2]-val_sens],np.uint8)
	# TR_MAX = np.array([mean_3[0]+hue_sens, mean_3[1]+sat_sens, mean_3[2]+val_sens],np.uint8)
	# tr_3=cv2.inRange(orig, TR_MIN, TR_MAX)
	tr_3=cv2.calcBackProject([frame],[0],rh3,[0,180],1)


	roi_4=orig[int(cols/2.3):int(cols/2.3)+SQ_SIZE, int(rows/4):int(rows/4+SQ_SIZE)]
	rh4 = cv2.calcHist([roi_4],[0], None, [180], [0, 180] )
	cv2.normalize(rh4,rh4,0,255,cv2.NORM_MINMAX)
	if(reset):
		mean_4=cv2.mean(roi_4)
	# TR_MIN = np.array([0, max(mean_4[1]-sat_sens,0), mean_4[2]-val_sens],np.uint8)
	# TR_MAX = np.array([mean_4[0]+hue_sens, mean_4[1]+sat_sens, mean_4[2]+val_sens],np.uint8)
	# tr_4=cv2.inRange(orig, TR_MIN, TR_MAX)
	tr_4=cv2.calcBackProject([frame],[0],rh4,[0,180],1)


	roi_5=orig[int(cols/2.5):int(cols/2.5)+SQ_SIZE, int(rows/2):int(rows/2+SQ_SIZE)]
	rh5 = cv2.calcHist([roi_5],[0], None, [180], [0, 180] )
	cv2.normalize(rh5,rh5,0,255,cv2.NORM_MINMAX)
	if(reset):
		mean_5=cv2.mean(roi_5)
	# TR_MIN = np.array([0, max(mean_5[1]-sat_sens,0), mean_5[2]-val_sens],np.uint8)
	# TR_MAX = np.array([mean_5[0]+hue_sens, mean_5[1]+sat_sens, mean_5[2]+val_sens],np.uint8)
	# tr_5=cv2.inRange(orig, TR_MIN, TR_MAX)
	tr_5=cv2.calcBackProject([frame],[0],rh5,[0,180],1)


	roi_6=orig[int(cols/8):int(cols/8)+SQ_SIZE, int(rows/2.7):int(rows/2.7+SQ_SIZE)]
	rh6 = cv2.calcHist([roi_6],[0], None, [180], [0, 180] )
	cv2.normalize(rh6,rh6,0,255,cv2.NORM_MINMAX)
	if(reset):
		mean_6=cv2.mean(roi_6)
	# TR_MIN = np.array([0, max(mean_6[1]-sat_sens,0), mean_6[2]-val_sens],np.uint8)
	# TR_MAX = np.array([mean_6[0]+hue_sens, mean_6[1]+sat_sens, mean_6[2]+val_sens],np.uint8)
	# tr_6=cv2.inRange(orig, TR_MIN, TR_MAX)
	tr_6=cv2.calcBackProject([frame],[0],rh6,[0,180],1)


	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	# dst=cv2.erode(dst, disc)
	# dst=cv2.dilate(dst, disc)
	# cv2.filter2D(dst,-1,disc,dst)
	#
	#res = cv2.bitwise_and(frame,thresh)
	#res = np.vstack((frame,thresh,res))
	#cv2.imshow("HS_", thresh)


	if(reset):
		reset=0
		

	tr = tr_1|tr_2|tr_3|tr_4|tr_5|tr_6

	ret,tr = cv2.threshold(tr_1,50,255,0)
	# tr = cv2.merge((thresh,thresh,thresh))

	ctr = cv2.medianBlur(tr, 9)
	tr = cv2.medianBlur(tr, 9)
	contours, hierarchy = cv2.findContours(ctr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# print len(contours)
	# cv2.drawContours(tr,contours,-1,(0,255,255),3)
	
	areas = [cv2.contourArea(c) for c in contours]
	if len(areas)>0:
		max_index = np.argmax(areas)
		cnt=contours[max_index]
		print max_index
		# cv2.drawContours(frame,cnt,-1,(0,255,255),3)
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


	cv2.imshow('HSV', frame) 
	cv2.imshow('HS_', tr)
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
		print "Sensing range", hue_sens, sat_sens, val_sens, ";", TR_MIN, TR_MAX
	elif c==ord('-'):
		if key == ord('h'):
			hue_sens-=1
		elif key==ord('s'):
			sat_sens-=1
		elif key==ord('v'):
			val_sens-=1
		print "Sensing range", hue_sens, sat_sens, val_sens, ";", TR_MIN, TR_MAX
	elif c==ord('z'):
		reset=1