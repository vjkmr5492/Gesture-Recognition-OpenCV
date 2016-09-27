import cv2
import cv2.cv as cv
import numpy as np
import math,inspect
from GestureFeature import GestureFeature


class GestureFeatureExtractor:
	def __init__(self,inputRawFeature):
		self.QUANTIZE_ANGLE = 10
		self.inputRawFeature=inputRawFeature
		self.n=len(inputRawFeature)
		self.extractedFeature=np.empty(shape=self.n,dtype=object)
		self.locationRelativeToCG=np.zeros(self.n)
		self.angleWithCG=np.zeros(self.n)
		self.angleWithInitialPt=np.zeros(self.n)
		self.angleWithEndPt=np.zeros(self.n)
		self.xMinyMinAngle=np.zeros(self.n)
		self.xMinyMaxAngle=np.zeros(self.n)
		self.xMaxyMinAngle=np.zeros(self.n)
		self.xMaxyMaxAngle=np.zeros(self.n)
		self.xMin=0.0
		self.xMax=0.0
		self.yMin=0.0
		self.yMax=0.0
		self.center=np.zeros(2)

	 	self.calculateFeatures()
	  	return

	def calculateFeatures(self):
		self.calculateCenterAndBounds()
		self.calculatePositionDistanceAngle()
		self.normalizeFeatures()
		self.composeFeatureVector()
		#self.printVectors()
		return


	def printVectors(self):
		print "locationRelativeToCG:"+str(self.locationRelativeToCG)
		print "angleWithCG:"+str(self.angleWithCG)
		print "angleWithInitialPt:"+str(self.angleWithInitialPt)
		print "angleWithEndPt:"+str(self.angleWithEndPt)
		return


	def calculateCenterAndBounds(self):
		self.xMin=0
		self.xMax=0
		self.yMin=0
		self.yMax=0
		sX=0
		sY=0
		for p in self.inputRawFeature:
			curX = p[0]
			curY = p[1]
			
			if curX < self.xMin:
				self.xMin = curX
			elif curX > self.xMax:
				self.xMax = curX
			if curY < self.yMin:
				self.yMin = curY
			elif curY > self.yMax:
				self.yMax = curY
			sX += curX
			sY += curY

		self.center[0] = sX / self.n
		self.center[1] = sY / self.n
		print "center",self.center
		return

	def divide(self,num, denom):
		if(denom == 0.0):
			return 0.0
		else:
			q=num/float(denom)
			#print "divide:",q,inspect.stack()[1]
			return q


	def getAngleYbyX(self,dy,dx):
		angleD = (math.degrees(math.atan(self.divide(dy, dx))) / self.QUANTIZE_ANGLE)
		#print angleD , ":" , math.floor(angleD)
		return math.ceil(angleD)


	def calculatePositionDistanceAngle(self):
		initXo=self.inputRawFeature[0][0]
		initYo=self.inputRawFeature[0][1]
		initXn=self.inputRawFeature[self.n-1][0]
		initYn=self.inputRawFeature[self.n-1][1]

		#print "F and L:",initXo,initYo,initXn,initYn

		for i in range(self.n-1):
			curPt = self.inputRawFeature[i]
			dxC = (curPt[0] - self.center[0])
			dyC = (curPt[1] - self.center[1])
			sqSum = np.power(dxC, 2) + np.power(dyC, 2)
			self.locationRelativeToCG[i] = np.power(sqSum,0.5)
			self.angleWithCG[i] = self.getAngleYbyX(dyC, dxC)
			#with successive
			self.angleWithInitialPt[i] = self.getAngleYbyX(curPt[1] - initYo, curPt[0] - initXo)
			self.angleWithEndPt[i] = self.getAngleYbyX(curPt[1] - initYn, curPt[0] - initXn)
			self.xMinyMinAngle[i] = self.getAngleYbyX(curPt[1] - self.yMin, curPt[0] - self.xMin)
			self.xMinyMaxAngle[i] = self.getAngleYbyX(curPt[1] - self.yMax, curPt[0] - self.xMin)
			self.xMaxyMinAngle[i] = self.getAngleYbyX(curPt[1] - self.yMax, curPt[0] - self.xMin)
			self.xMaxyMaxAngle[i] = self.getAngleYbyX(curPt[1] - self.yMax, curPt[0] - self.xMax)


		return

	def normalizeFeatures(self):
		#normalize so that input lies in a specific range
		print "self.locationRelativeToCG before normalize"
		for i in self.locationRelativeToCG:
			print i

		maxLoc = max(self.locationRelativeToCG)
		minLoc = min(self.locationRelativeToCG)

		#cross check the range
		for i in range(self.n):
			self.locationRelativeToCG[i] = self.divide(self.locationRelativeToCG[i] - minLoc, maxLoc - minLoc)

		print "self.locationRelativeToCG after normalize"
		for i in self.locationRelativeToCG:
			print i


	def composeFeatureVector(self):
		for i in range(self.n):
			self.extractedFeature[i] = GestureFeature()
			self.extractedFeature[i].angleWithCG = self.angleWithCG[i]
			self.extractedFeature[i].angleWithInitialPt = self.angleWithInitialPt[i]
			self.extractedFeature[i].locationRelativeToCG = self.locationRelativeToCG[i]
			self.extractedFeature[i].angleWithEndPt = self.angleWithEndPt[i]
			self.extractedFeature[i].xMaxyMaxAngle = self.xMaxyMaxAngle[i]
			self.extractedFeature[i].xMaxyMinAngle = self.xMaxyMinAngle[i]
			self.extractedFeature[i].xMinyMaxAngle = self.xMinyMaxAngle[i]
			self.extractedFeature[i].xMinyMinAngle = self.xMinyMinAngle[i]

		return
		


if __name__=='__main__':
	x=GestureFeatureExtractor([[87,114],[89,111],[93,109],[98,104],[105,96],[109,91],[114,86],[119,80],[123,76],[128,71],[130,70],[131,68],[131,74],[131,84],[131,92],[131,103],[131,113],[131,122],[131,128],[131,133],[131,136],[131,139],[132,143],[133,144],[135,144],[138,144],[142,142],[149,137],[156,131],[162,126],[170,119],[176,114],[180,111],[182,108],[183,108],[184,109],[184,117],[185,129],[185,143],[185,155],[185,165],[185,173],[186,181],[186,186],[186,187],[187,187],[188,187],[191,187],[197,186],[202,180],[208,175],[217,167],[225,161],[234,152],[240,146],[245,140],[249,136],[250,133],[254,130],[254,129],[255,129],[255,134],[257,144],[259,153],[260,163],[263,172],[265,180],[268,187],[270,191],[271,193],[271,194],[272,194],[275,194],[279,193],[285,188],[291,182],[300,176],[309,168],[318,160],[325,154],[330,149],[332,146],[333,146]])
	#for b in x.extractedFeature:
	#	print b.getFeatureVector()
