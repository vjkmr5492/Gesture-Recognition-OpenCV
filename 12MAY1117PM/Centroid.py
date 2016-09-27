from Points import Points
from pprint import pprint

class Centroid(Points):
	


	def __init__(self, co):

		self.distortion=0
		self.pts=[]
		self.total_pts=0
		Points.__init__(self, co)
		# for i in self.pts:
		# 	pprint(vars(i))
		self.total_pts=0
		# self.coordinates=co
		# self.dimension=len(co)


	def getPoint(self, i):
		return self.pts[i]

	def getNumPoints(self):
		return self.total_pts

	def remove(self, pt, dist):
		i=0
		temppt=-1
		for ept in self.pts:
			# pprint(vars(pt))
			if Points.areEqual(ept,pt) == True:
				temppt=i
				break
			i+=1
		if temppt!=-1:
			del self.pts[temppt]
			self.distortion-=dist
			self.total_pts-=1
		else:
			print "Cannot remove, not found"

	def add(self, pt, dist):
		self.total_pts+=1
		self.pts.append(pt)
		self.distortion+=dist

	def update(self):
		su=[]
		su=[0]*self.dimension
		for i in range(len(self.pts)):
			s=self.pts[i]
			for k in range(self.dimension):
				su[k]+=s.getCo(k)
		# print su
		for i in range(self.dimension):
			self.setCo(i, su[i]/float(self.total_pts))
			self.pts=[]
			# print "DV", self.total_pts
		self.total_pts=0
		self.distortion=0

	def getDistortion(self):
		return self.distortion
