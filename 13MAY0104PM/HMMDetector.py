import cv2
import cv2.cv as cv
import numpy as np
import os, pickle
import time, pprint
from GestureFeatureExtractor import GestureFeatureExtractor as GFE

from mycython import HiddenMarkov
from Codebook import Codebook
from Points import Points
from Centroid import Centroid


def save_object(obj, filename):
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
		
def load_object(filename):
	with open(filename, 'rb') as input:
		return pickle.load(input)

def directory(name):
	if not os.path.exists(name):
		os.makedirs(name)
		
def detect(boxes):
		ml=[]
		cb=Codebook()
		cb.load()
		ob = GFE(boxes)
		qv=[]
		vv=[]
		for i in range(len(boxes)-1):
			qv.append(  Points(ob.extractedFeature[i].getNFeatureVector())  )
		X=np.vstack(qv).squeeze()
		qt = cb.quantize(X)
		# print qt
		
		# qt=[6,6,7,7,7,7,2,2,3,3,3,3,3,5,5,5,20,20,20,21,21,21,22,22,22,22,22,19,19,19,19,19,19,19,19,19,19,19,19,18,18,18,8,8,9,9,9,11,11,14,14,14,14,14,29,29,29,29,30,30,27,25,24,15]
		dirs = [name for name in os.listdir(".") if os.path.isdir(os.path.join(".", name)) and name!='data' and name!='build']
		for md in dirs:
			x=HiddenMarkov(4,64)
			x.initFromName("./"+md+"/"+md+".hmm")
			ml.append(x)
			vv.append(x.viterbi(np.array(qt)))

			# print md, x.viterbi(np.array(qt))
			x=None
		print dirs[vv.index(max(vv))]

def gen_cb():
	print "Generating Codebook..."
	vector=[]
	dirs = [name for name in os.listdir(".") if os.path.isdir(os.path.join(".", name)) and name!='data' and name!='build']
	for diri in dirs:
		
		print "Working for "+diri
		raws = [name for name in os.listdir("./"+diri) if name.endswith(".raw")]
		for name in raws:
			print "Processing file "+name
			rl =  load_object("./"+diri+"/"+name)
			ob = GFE(rl)
			for i in range(len(rl)-1):
				vector.append(ob.extractedFeature[i].getNFeatureVector())
	X=np.vstack(vector).squeeze()
	# print X
	directory("data")
	c=Codebook()
	c.genCB(X)
	c.saveToFile()

def train():
		cb=Codebook()
		cb.load()
		dirs = [name for name in os.listdir(".") if os.path.isdir(os.path.join(".", name)) and name!='data' and name!='build']
		
		mkv=HiddenMarkov(4, 64)
		for tname in dirs:
			tmp=[]    
			raws = [name for name in os.listdir("./"+tname) if name.endswith(".raw")]
			for name in raws:
				rl =  load_object("./"+tname+"/"+name)
				ob = GFE(rl)
				fqv=[]
				for i in range(len(rl) - 1):
					fqv.append(  Points(ob.extractedFeature[i].getNFeatureVector())  )
				tm=cb.quantize(fqv)
				# print "Training seq", tm
				tmp.append(tm)
				# if name==raws[0]:
				# 	mkv.setITrainSeq(np.array(tm))
				# else:
				# 	mkv.setTrainSeq(np.array(tm))
			
			print "Starting training for ", tname
			mkv.setTrainSeq(tmp)
			print "training", tname
			mkv.train()
			print "Saving", tname
			mkv.save("./"+tname+"/"+tname+".hmm")
			# print "OP", mkv.output
			# print "TR", mkv.transition
		print "Training done"