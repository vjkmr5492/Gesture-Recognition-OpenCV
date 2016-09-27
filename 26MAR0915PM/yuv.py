import cv2.cv as cv
import cv2
import numpy as np
first=0
found=0
maskbox=None
SQ_SIZE=20
track=None
oldpt=None
cap = cv2.VideoCapture(0)
cv.NamedWindow('YUV', cv.CV_WINDOW_AUTOSIZE)
cv.NamedWindow('SKIN', cv.CV_WINDOW_AUTOSIZE)

while True:
	_, frame = cap.read()
	orig=frame
	rows=frame.shape[0]
	cols=frame.shape[1]
	if track is None:
		track=np.ones((rows,cols,3), np.uint8)
	# if maskbox is None:
	# 	maskbox=np.ones((rows, cols, 1), np.uint8)
	# else:
	# 	frame=cv2.bitwise_and(frame , frame, mask= maskbox)
	
	frame=cv2.GaussianBlur(frame,(5,5), 15)
	if first==0:
		bg=frame
		first=1

	yuv=cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
	# skin=cv2.inRange(yuv, (0,133,111), (255,150,133))
	skin=cv2.inRange(yuv, (51,130,115), (100,203,131))
	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

	skin=cv2.erode(skin, kernel)
	skin=cv2.dilate(skin, kernel)
	hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	hsv[:,:,2]=0

	cv2.rectangle(frame, (int(rows/3),int(cols/2.5)), (int(rows/3+SQ_SIZE),int(cols/2.5)+SQ_SIZE), (0,255,255))
	roi_1=orig[int(cols/2.5):int(cols/2.5)+SQ_SIZE, int(rows/3):int(rows/3+SQ_SIZE)]
	# print cv2.mean(roi_1)

	gbgp=cv2.cvtColor((bg-frame), cv2.COLOR_BGR2GRAY)
	gbg=cv2.inRange(gbgp, (0),  (80))


	hsv_skin=cv2.inRange(hsv, (0,50,0), (50,150,10))
	

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	skin = cv2.dilate(skin,kernel)
	skin = cv2.erode(skin,kernel)
	skin=cv2.medianBlur(skin, 7)

	hsv_skin=cv2.medianBlur(hsv_skin, 3)

	ctr=np.array(skin, copy=True)
	ctr = cv2.medianBlur(skin, 1)
	tr = cv2.medianBlur(skin, 1)
	contours, hierarchy = cv2.findContours(ctr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	areas = [cv2.contourArea(c) for c in contours]
	if len(areas)>0:
		max_index = np.argmax(areas)
		maskbox=np.zeros((rows, cols, 1), np.uint8)
		cnt=contours[max_index]
		area=areas[max_index]
		moments=cv2.moments(cnt)
		if areas[max_index]>5000:
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			hull=cv2.convexHull(cnt)
			# cv2.fillPoly(frame, hull, (0,0,0))
			cv2.drawContours(frame,[hull], -1, (0,0,0), 4)
			# maskbox[max(0,y-20):min(y+h+20, rows),max(x-20, 0):min(cols,x+w+20)]=1
			cx = int(moments['m10']/moments['m00'])         # cx = M10/M00
			cy = int(moments['m01']/moments['m00'])
			
			if oldpt is None:
				cv2.circle(track,(cx,cy),5,(0,0,255),-1) 
				oldpt=(cx,cy)
			else:
				cv2.line(track, oldpt , (cx, cy) , (0,0,255), 8)
				oldpt=(cx,cy)
			"""cv2.circle(skin,(cx,cy+25),int(h/4),(0,0,0),-1) 
			handregion=skin[y:y+h, x:x+w]
			handregion[cy-100:] = 0
			frame[y:y+h, x:x+w]=cv2.cvtColor(handregion, cv2.COLOR_GRAY2BGR)
			fingers, hierarchy = cv2.findContours(handregion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
			fareas = [cv2.contourArea(f) for f in fingers]
			if len(fingers)>0:
				nf=0;
				for i in range(0, len(fareas)):
					if fareas[i]>700:
						fx,fy,fw,fh = cv2.boundingRect(fingers[i])
						if fy+fh<cy and fw!=w:
							nf=nf+1
							cv2.rectangle(frame,(x+fx,y+fy),(x+fx+fw,y+fy+fh),(0,255,0),2)
						elif fw==w:
							nf=0
			
				print nf
			"""


		# for i in range(0,len(contours)):
		# 	c=contours[i]
			# if cv2.contourArea(c)<areas[max_index] and i!=max_index:
				# xb,yb,wb,hb = cv2.boundingRect(c)
				# cv2.rectangle(frame,(xb,yb),(xb+wb,yb+hb),(0,255,0),2)
				# maskbox[max(0,yb):min(yb+hb, rows),max(xb, 0):min(cols,xb+wb)]=0
				# cv2.drawContours(maskbox,contours, i, (0,0,0))
		
		# moments = cv.Moments(skin, 0)
		# x = cv.GetSpatialMoment(moments, 1, 0)/area
		# y = cv.GetSpatialMoment(moments, 0, 1)/area
		# cv.Circle(skin, (x, y), 2, (255, 255, 255), 20)  




	cv2.imshow('YUV', track | frame) 
	cv2.imshow('SKIN',   skin  )
	c = cv.WaitKey(1)
	if c == 27 : 
		break
	elif c==ord('r'):
		track=np.ones((rows,cols,3), np.uint8)

