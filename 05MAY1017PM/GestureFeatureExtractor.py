import cv2
import cv2.cv as cv
import numpy as np
import math
from GestureFeature import GestureFeature


class GestureFeatureExtractor:
	def __init__(self,points):
		self.points=points
		self.n=len(points)
		self.QUANTIZE_ANGLE = 10
		self.CHAIN_CODE_NUM_DIRECTIONS = 16
		self.extractedFeature=np.empty(shape=self.n-1,dtype=object)
		self.locationRelativeToCG=np.zeros(self.n-1)
		self.distanceBetweenSuccessivePts=np.zeros(self.n-1)
		self.angleWithCG=np.zeros(self.n-1)
		self.angleWithInitialPt=np.zeros(self.n-1)
		self.angleWithEndPt=np.zeros(self.n-1)
		self.xMinyMinAngle=np.zeros(self.n-1)
		self.xMinyMaxAngle=np.zeros(self.n-1)
		self.xMaxyMinAngle=np.zeros(self.n-1)
		self.xMaxyMaxAngle=np.zeros(self.n-1)
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
		print "distanceBetweenSuccessivePts:"+str(self.distanceBetweenSuccessivePts)
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
		for p in self.points:
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
		return

	def divide(self,num, denom):
		if(denom == 0.0):
			return 0.0
		else:
			return num/float(denom)


	def getAngleYbyX(self,dy,dx):
		angleD = (math.degrees(math.atan(self.divide(dy, dx))) / self.QUANTIZE_ANGLE)
		#print angleD , ":" , math.floor(angleD)
		return (math.ceil(angleD))


	def calculatePositionDistanceAngle(self):
		initXo=self.points[0][0]
		initYo=self.points[0][1]
		initXn=self.points[self.n-1][0]
		initYn=self.points[self.n-1][1]

		for i in range(self.n-1):
			curPt = self.points[i]
			dxC = (curPt[0] - self.center[0])
			dyC = (curPt[1] - self.center[1])
			sqSum = math.pow(dxC, 2) + math.pow(dyC, 2)
			self.locationRelativeToCG[i] = math.sqrt(sqSum)
			self.angleWithCG[i] = self.getAngleYbyX(dyC, dxC)
			#with successive
			p2 = self.points[i + 1]
			dxSu = (curPt[0] - p2[0])
			dySu = (curPt[1] - p2[1])
			self.distanceBetweenSuccessivePts[i] = math.sqrt(math.pow(dxSu, 2) + math.pow(dySu, 2))
			self.angleWithInitialPt[i] = self.getAngleYbyX(curPt[1] - initYo, curPt[0] - initXo)
			self.angleWithEndPt[i] = self.getAngleYbyX(curPt[1] - initYn, curPt[0] - initXn)
			self.xMinyMinAngle[i] = self.getAngleYbyX(curPt[1] - self.yMin, curPt[0] - self.xMin)
			self.xMinyMaxAngle[i] = self.getAngleYbyX(curPt[1] - self.yMax, curPt[0] - self.xMin)
			self.xMaxyMinAngle[i] = self.getAngleYbyX(curPt[1] - self.yMax, curPt[0] - self.xMin)
			self.xMaxyMaxAngle[i] = self.getAngleYbyX(curPt[1] - self.yMax, curPt[0] - self.xMax)


		return

	def normalizeFeatures(self):
		#normalize so that input lies in a specific range
		maxLoc = max(self.locationRelativeToCG)
		maxDist = max(self.distanceBetweenSuccessivePts)
		minLoc = min(self.locationRelativeToCG)
		minDist = min(self.distanceBetweenSuccessivePts)

		#cross check the range
		for i in range(self.n-1):
			self.locationRelativeToCG[i] = self.divide(self.locationRelativeToCG[i] - minLoc, maxLoc - minLoc)
			self.distanceBetweenSuccessivePts[i] = self.divide(self.distanceBetweenSuccessivePts[i] - minDist, maxDist - minDist)

		return

	#def quantizeAngle(self):

	def composeFeatureVector(self):
		for i in range(self.n-1):
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
	x=GestureFeatureExtractor([[1,2],[2,2],[5,3]])

	for i in range(2):
		print x.extractedFeature[i].getFeatureVector()
