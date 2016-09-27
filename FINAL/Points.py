
class Points(object):
	def __init__(self, arg):
		self.coordinates=arg
		self.dimension=len(arg)

	def getAllCo(self):
		return self.coordinates

	def getCo(self, i):
		return self.coordinates[i]

	def setCo(self, i, val):
		self.coordinates[i]=val

	def changeCo(self, newco):
		self.coordinates=newco

	def getDimension(self):
		return self.dimension

	def areEqual(a,b):
		for i in range(len(a.coordinates)):
			if a.coordinates[i]!=b.coordinates[i]:
				return False
		return True

		
