import cv2.cv as cv
import cv2
import numpy as np
import math




#handle tracking outside
def angle(p1, p2): 
	xDiff = p2[0]-p1[0]
	yDiff= p2[1]-p1[1] 
	return math.degrees(math.atan2(yDiff,xDiff)) 
def dist(p1, p2):
	distn = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
	return distn

def clean_cos(c_ang):
	return  min(1, max(c_ang, -1))

def ng3(c, p0, p1):
	p0c = math.sqrt(math.pow(c[0]-p0[0],2)+math.pow(c[1]-p0[1],2))
	p1c = math.sqrt(math.pow(c[0]-p1[0],2)+math.pow(c[1]-p1[1],2))
	p0p1 = math.sqrt(math.pow(p1[0]-p0[0],2)+math.pow(p1[1]-p0[1],2))
	# if ((p1c*p1c+p0c*p0c-p0p1*p0p1)/(2*p1c*p0c)) >= 1.0 or ((p1c*p1c+p0c*p0c-p0p1*p0p1)/(2*p1c*p0c))<=-1.0:
		# print ((p1c*p1c+p0c*p0c-p0p1*p0p1)/(2*p1c*p0c))
	if 2*p1c*p0c!=0.0:
		return ((  (p1c*p1c+p0c*p0c-p0p1*p0p1)/(2*p1c*p0c) ))
	else:
		return 0

def zcross(r0, r1, r2):
	vx1=r0[0]-r1[0];
	vy1=r0[1]-r1[1];
	vx2=r0[0]-r2[0];
	vy2=r0[1]-r2[1];
	return (vx1*vy2 - vx2*vy1);


def curve(r0, r1, r2):
	vx1=r0[0]-r1[0]
	vy1=r0[1]-r1[1]
	vx2=r0[0]-r2[0]
	vy2=r0[1]-r1[1]
	return (vx1*vx2 + vy1*vy2)/math.sqrt((vx1*vx1 + vy1*vy1)*(vx2*vx2 + vy2*vy2));




#retruns Contour,#fingers and a finger Position
def getContourFingersAndDefects(frame,skin):
	highestFinger=[[500,500]]
	fingerCount=0
	fingers=[]
	skin=cv2.medianBlur(skin, 7)
	ctr=np.array(skin, copy=True)
	ctr = cv2.medianBlur(skin, 1)
	tr = cv2.medianBlur(skin, 1)
	contours, hierarchy = cv2.findContours(ctr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	areas = [cv2.contourArea(c) for c in contours]
	if len(areas)>0:
		max_index = np.argmax(areas)
		cnt=contours[max_index]

		area=areas[max_index]
		moments=cv2.moments(cnt)
		if areas[max_index]>5000:
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.drawContours(frame,cnt, -1,(255,255, 0),3)
			# print "ST"

			boxx, boxy= x+w/2, y+h/2
			

			step=3
			dst=20
			cx = int(moments['m10']/moments['m00'])         # cx = M10/M00
			cy = int(moments['m01']/moments['m00'])
			

			for i in range(0, len(cnt), step):
			# if i>=0:
				p1=cnt[(i-dst)%len(cnt)]
				c= cnt[i]
				p2= cnt[(i+dst)%len(cnt)]
				cos0=(ng3(c[0], p1[0], p2[0]))
				zc = zcross(c[0], p1[0], p2[0])
				
				if cos0>0.2:
					cos1 = (ng3(cnt[(i-step)%len(cnt)][0],cnt[(i-step-dst)%len(cnt)][0], cnt[(i-step+dst)%len(cnt)][0] ))
					cos2 = (ng3(cnt[(i+step)%len(cnt)][0],cnt[(i+step-dst)%len(cnt)][0], cnt[(i+step+dst)%len(cnt)][0] ))
					val = max(cos0, cos1, cos2)
					if val==cos0:
						if zc<0:
							cv2.circle(frame,(c[0][0], c[0][1]),5,[255,0,255],-1)
							cv2.putText(frame,"D", (c[0][0], c[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
						else:
							if c[0][1]< boxy:
								cv2.circle(frame,(c[0][0], c[0][1]),5,[0,255,255],-1)
								cv2.putText(frame,"F", (c[0][0], c[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
								fingerCount+=1
								fingers.append(c)
		# print "FINGERS", fingers
	#find max
		

		for x in range(0,len(fingers)):
			if fingers[x][0][1] < highestFinger[0][1]:
				highestFinger=fingers[x]

	return frame,fingerCount,highestFinger[0]


if __name__=='__main__':
	frame=cv2.imread('hand.png')
	skin=cv2.inRange(frame,(244,244,244),(255,255,255))

	frame,n,pt=getContourFingersAndDefects(frame,skin)
	print n,pt
	while True:
		if cv.WaitKey(-1) == 27:
			break;
		cv2.circle(frame,(pt[0],pt[1]),5,(0,0,255),-1) 
		cv2.imshow('frame',frame)
		

				
				
				
			
			  




	
	




