import cv2
import cv2.cv as cv
import numpy as np


class GestureFeature:
	def __init__(self):
		self.locationRelativeToCG=0
		self.angleWithCG=0
		self.angleWithInitialPt=0
		self.angleWithEndPt=0
		self.xMinyMinAngle=0
		self.xMinyMaxAngle=0
		self.xMaxyMinAngle=0
		self.xMaxyMaxAngle=0
		return

	def getFeatureVector(self):
		Featurevector = np.array([self.locationRelativeToCG,
						self.angleWithCG,
						self.angleWithInitialPt, 
						self.angleWithEndPt,
						self.xMaxyMaxAngle, 
						self.xMaxyMinAngle,
						self.xMinyMaxAngle,
						self.xMinyMinAngle])
		return Featurevector

	def getNFeatureVector(self):
		Featurevector = [self.locationRelativeToCG,
						self.angleWithCG,
						self.angleWithInitialPt, 
						self.angleWithEndPt,
						self.xMaxyMaxAngle, 
						self.xMaxyMinAngle,
						self.xMinyMaxAngle,
						self.xMinyMinAngle]
		return Featurevector


if __name__=='__main__':
	print GestureFeature().getFeatureVector()