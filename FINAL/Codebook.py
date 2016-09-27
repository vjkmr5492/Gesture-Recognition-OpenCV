import numpy as np
from Centroid import Centroid
from Points import Points
import pickle, math
from pprint import pprint

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


class Codebook(object):
    	 
    

    def __init__(self):
        """ generated source for method __init___0 """
        # print "CB Created"

        self.SPLIT = 0.01 	 
    	self.MIN_DISTORTION = 0.1 
    	self.codebook_size = 64
    	self.dimension=0 
    	self.centroids = []
    	self.pt=[]
        

    def genCB(self, tmpPt):

    	for i in tmpPt:
        	self.pt.append(Points(i))
        if len(self.pt) >= self.codebook_size:
            self.dimension = self.pt[0].getDimension()

            self.initialize()

        else:
            print "err: not enough training points"


    def load(self):
        """ generated source for method __init___1 """
        # Load from file
        c=load_object("./data/codebook.cbk")
        self.dimension=c.dimension
        self.centroids=c.centroids



        # print len(c.centroids)
        # for ec in c.centroids:
        	# pprint(vars(ec))
            # print len(ec.pts)
            # for ep in ec.pts:
            #     print "p", ep.coordinates

    def initialize(self):
        """ generated source for method initialize """
        distortion_before_update = 0
        distortion_after_update = 0
        self.centroids = []
        origin = Centroid([0]*self.dimension)
        self.centroids.append(origin)
        for i in range(len(self.pt)):
        	# print "CENTROID", len(Centroid.pts)
        	self.centroids[0].add(self.pt[i], 0)


        self.centroids[0].update()
        
        
        """
        for i in self.centroids:
        	pprint(vars(i))
        return
        """

        while len(self.centroids)<self.codebook_size:

			# print "B4 SPLIT ", i, "UPDATE"
			# for z in self.centroids:
			# 	pprint(vars(z))
			# 	for px in z.pts:
			# 		print px.coordinates
			# 	print len(z.pts)
			# print "B4 END"
			

			self.split()

			
			# print "AF SPLIT ", i, "UPDATE"
			# for z in self.centroids:
			# 	pprint(vars(z))
			# 	for px in z.pts:
			# 		print px.coordinates
			# 	print len(z.pts)
			# print "AF END"

			self.groupPtoC()

			# print "P2C ", i, "UPDATE"
			# for z in self.centroids:
			# 	pprint(vars(z))
			# 	for px in z.pts:
			# 		print px.coordinates
			# 	print len(z.pts)
			# print "P2C END"
			

			
				# 
			while True:
				i=0
				while i<len(self.centroids):
					distortion_before_update += self.centroids[i].getDistortion()

					# print "BFR ", i, "UPDATE"
					# for z in self.centroids:
					# 	pprint(vars(z))
					# 	for px in z.pts:
					# 		print px.coordinates
					# 	print len(z.pts)
					# print "BFR END"
					
					self.centroids[i].update()
					

					# print "AFT", i, "UPDATE"
					# for z in self.centroids:
					# 	pprint(vars(z))
					# 	# for px in z.pts:
					# 	# 	print px.coordinates
					# 	print len(z.pts)
					# print "AFT END"

					i += 1

				

				self.groupPtoC()

				

				i=0
				while i<len(self.centroids):
					distortion_after_update += self.centroids[i].getDistortion()
					i += 1
				if not ((abs(distortion_after_update - distortion_before_update) < self.MIN_DISTORTION)):
					break

    def saveToFile(self):
    	c=Codebook()
    	c.dimension=self.dimension
    	c.centroids=self.centroids

    	for ec in c.centroids:
        	print ec.coordinates

    	save_object(c, "./data/codebook.cbk")



        # Save To File

    def a2d(self, a,b):
		twod_list = []      
		for i in range (0, a):     
			# new = []                
   #  			for j in range (0, b): 
   #      			new.append(0)     
   #  		twod_list.append(new)
   			twod_list.append(self.a1d(b))
		return twod_list
    
    def a1d(self, a):
    	new=[]
    	for i in range (0, a):     
    		new.append(0)  
    	return new

    def split(self):

		# print "LEN",len(self.centroids[0].pts)

		""" generated source for method split """
		print "Size is now ", str(2 * len(self.centroids))
		temp = self.a1d(len(self.centroids)*2)
		tCo = 0
		i = 0

		while i<len(temp):

			tCo = self.a2d(2, self.dimension)
			j=0
			while j < self.dimension:
				
				tCo[0][j] = self.centroids[i / 2].getCo(j) * (1 + self.SPLIT)
				# print "tCo[0]["+str(j)+"]", tCo[0][j],  self.centroids[i / 2].getCo(j)
				j += 1
			temp[i] = Centroid(tCo[0])
			# print temp[i].pts
			# print "LEN",len(temp[0].pts)
			j=0
			while j < self.dimension:
				tCo[1][j] = self.centroids[i / 2].getCo(j) * (1 - self.SPLIT)
				# print "tCo[1]["+str(j)+"]", tCo[1][j],  self.centroids[i / 2].getCo(j)
				j += 1
			temp[i + 1] = Centroid(tCo[1])
			i += 2
		

		self.centroids = temp
		
		# print "SPLITFN START ", len(self.centroids), "SIZED"
		# for z in self.centroids:
		# 	pprint(vars(z))
		# 	for px in z.pts:
		# 		print px.coordinates
		# 	print len(z.pts)
		# print "SPLITFN END"
			
        


        

    def quantize(self, pts):
        """ generated source for method quantize """
        output = self.a1d(len(pts))
        i = 0
        while i<len(pts):
            output[i] = self.closestCentroidToPoint(pts[i])
            i += 1
        return output

    def getDistortion(self, pts):
		""" generated source for method getDistortion """
		dist = 0
		i = 0
		while i<len(pts):
				idx=self.closestCentroidToPoint(pts[i])
				d=self.getDistance(pts[i], self.centroids[idx])
				dist += d
				i += 1
		return dist

    def closestCentroidToPoint(self, pt):
        """ generated source for method closestCentroidToPoint """
        tmp_dist = 0
        lowest_dist = 0
        lowest_index = 0
        i = 0
        while i<len(self.centroids):
            tmp_dist = self.getDistance(pt, self.centroids[i])
            if tmp_dist < lowest_dist or i == 0:
                lowest_dist = tmp_dist
                lowest_index = i
            i += 1
        return lowest_index

    def closestCentroidToCentroid(self, c):
        """ generated source for method closestCentroidToCentroid """
        tmp_dist = 0
        lowest_dist = float("inf")
        lowest_index = 0
        i = 0
        while i<len(self.centroids):
            tmp_dist = self.getDistance(c, self.centroids[i])
            if tmp_dist < lowest_dist and self.centroids[i].getNumPoints() > 1:
                lowest_dist = tmp_dist
                lowest_index = i
            i += 1
        return lowest_index

    def closestPoint(self, c1, c2):
        """ generated source for method closestPoint """
        tmp_dist = 0
        lowest_dist = self.getDistance(c2.getPoint(0), c1)
        lowest_index = 0
        i = 1
        while i < c2.getNumPoints():
            tmp_dist = self.getDistance(c2.getPoint(i), c1)
            if tmp_dist < lowest_dist:
                lowest_dist = tmp_dist
                lowest_index = i
            i += 1
        return lowest_index

    def groupPtoC(self):
		""" generated source for method groupPtoC """
		i = 0
		while i<len(self.pt):
        		idx=self.closestCentroidToPoint(self.pt[i])
        		self.centroids[idx].add(self.pt[i], self.getDistance(self.pt[i], self.centroids[idx]))
        		i += 1
		i = 0
		while i<len(self.centroids):
			if self.centroids[i].getNumPoints() == 0:
				idx=self.closestCentroidToCentroid(self.centroids[i])
				ci=self.closestPoint(self.centroids[i], self.centroids[idx])
				closestPt=self.centroids[idx].getPoint(ci)
				self.centroids[idx].remove(closestPt, self.getDistance(closestPt, self.centroids[idx]))
				self.centroids[i].add(closestPt, self.getDistance(closestPt, self.centroids[i]))
			i += 1

    def getDistance(self, tPt, tC):
        """ generated source for method getDistance """
        distance = 0
        temp = 0
        i = 0
        while i < self.dimension:
            temp = tPt.getCo(i) - tC.getCo(i)
            distance += temp * temp
            i += 1
        distance = np.power(distance, 0.5)
        

        return distance

