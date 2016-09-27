import pickle, os, math, time
from pprint import pprint
from Centroid import Centroid
from Points import Points
from Codebook import Codebook
from HiddenMarkov import HiddenMarkov
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def pb(ob):
	pprint(vars(ob))


vs=[4, 4, 6, 7, 7, 44, 44, 44, 44, 45, 45, 46, 46, 47, 47, 47, 47, 47, 34, 34, 35, 32, 33, 37, 37, 37, 36, 36, 36, 36, 36, 30, 28, 25, 24, 26, 20, 21, 22, 22, 22, 22, 22, 22, 31, 31, 31, 3, 3, 2, 1, 0, 1, 2, 23, 23]
md="Line"
x=HiddenMarkov(4,64)
x.initFromName("./"+md+"/"+md+".hmm")
print x.viterbi(vs)









