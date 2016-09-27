import pickle, os, math
from pprint import pprint
from Centroid import Centroid
from Points import Points
from Codebook import Codebook


def pb(ob):
	pprint(vars(ob))

x=[[1,2,3],[4,5,6],[7,8,10],[10,11,12],[13,14,15], [16,17,18]]

c=Codebook()

c.genCB(x)
"""
print x
"""


ct=Centroid([1,2,3])
print "X is ", x








