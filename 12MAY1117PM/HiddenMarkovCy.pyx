# import math
from libc cimport math
import numpy
import random
from HMMModel import HMMModel
import pickle

class HiddenMarkov(object):
       
    MIN_PROBABILITY = 0.00000000001
  
   
   
    delta = 2
   
    def viterbi(self, testSeq):
        """ generated source for method viterbi """
        self.setObSeq(testSeq)
        phi = numpy.zeros(shape=(int(self.len_obSeq),int(self.num_states)))
        self.psi = numpy.zeros(shape=(int(self.len_obSeq),int(self.num_states)))
        self.q = numpy.zeros(shape=(int(self.len_obSeq)))
        i = 0
        while i < self.num_states:
            temp = self.pi[i]
            if temp == 0:
                temp = self.MIN_PROBABILITY
            phi[0][i] = numpy.log(temp) + numpy.log(self.output[i][self.currentSeq[0]])
            self.psi[0][i] = 0
            i += 1
        t = 1
        while t < self.len_obSeq:
            j=0
            while j < self.num_states:
                maxi=phi[t-1][0]+ numpy.log(self.transition[0][j])
                temp=0
                index=0
                i=1
                while i < self.num_states:
                    temp = phi[t - 1][i] + numpy.log(self.transition[i][j])
                    if temp > maxi:
                        maxi = temp
                        index = i
                    i += 1
                phi[t][j] = maxi + numpy.log(self.output[j][self.currentSeq[t]])
                self.psi[t][j] = index
                j += 1
            t += 1
        maxi = phi[self.len_obSeq - 1][0]
        temp = 0
        index = 0
        i = 1
        while i < self.num_states:
            temp = phi[self.len_obSeq - 1][i]
            if temp > maxi:
                maxi = temp
                index = i
            i += 1
        self.q[self.len_obSeq - 1] = index
        t = self.len_obSeq - 2
        while t >= 0:
            self.q[t] = self.psi[t + 1][self.q[t + 1]]
            t -= 1
        return maxi

    def rescaleBeta(self, t):
        """ generated source for method rescaleBeta """
        i = 0
        while i < self.num_states:
            self.beta[t][i] *= self.scaleFactor[t]
            i += 1

    def rescaleAlpha(self, t):
        """ generated source for method rescaleAlpha """
        i = 0
        while i < self.num_states:
            self.scaleFactor[t] += self.alpha[t][i]
            i += 1
        self.scaleFactor[t] = 1 / self.scaleFactor[t]
        i = 0
        while i < self.num_states:
            self.alpha[t][i] *= self.scaleFactor[t]
            i += 1

    def getProbability(self, testSeq):
        """ generated source for method getProbability """
        self.setObSeq(testSeq)
        temp = self.computeAlpha()
        return temp

    def computeAlpha(self):
        """ generated source for method computeAlpha """
        cdef double probability = 0
        cdef int t = 0
        cdef double sumi=0
        cdef int i=0
        cdef int j=0
        while t < self.len_obSeq:
            self.scaleFactor[t] = 0
            t += 1
        i = 0
        while i < self.num_states:
            self.alpha[0][i] = self.pi[i] * self.output[i][self.currentSeq[0]]
            i += 1
        self.rescaleAlpha(0)
        t = 0
        while t < self.len_obSeq - 1:
            j=0
            while j < self.num_states:
                sumi=0
                i=0
                while i < self.num_states:
                    sumi += self.alpha[t][i] * self.transition[i][j]
                    i += 1
                self.alpha[t + 1][j] = sumi * self.output[j][self.currentSeq[t + 1]]
                j += 1
            self.rescaleAlpha(t + 1)
            t += 1
        i = 0
        while i < self.num_states:
            probability += self.alpha[self.len_obSeq - 1][i]
            i += 1
        probability = 0
        t = 0
        while t < self.len_obSeq:
            probability += math.log(self.scaleFactor[t])
            t += 1
        return -probability

    def computeBeta(self):
        """ generated source for method computeBeta """
        cdef int i = 0
        cdef int t=0
        cdef int j=0
        while i < self.num_states:
            self.beta[self.len_obSeq - 1][i] = 1
            i += 1
        self.rescaleBeta(self.len_obSeq - 1)
        t = self.len_obSeq - 2
        while t >= 0:
            i=0
            while i < self.num_states:
                j=0
                while j < self.num_states:
                    self.beta[t][i] += self.transition[i][j] * self.output[j][self.currentSeq[t + 1]] * self.beta[t + 1][j]
                    j += 1
                i += 1
            self.rescaleBeta(t)
            t -= 1

    

    
    def setTrainSeq(self, trainSeq):
        """ generated source for method setTrainSeq_0 """
        self.num_obSeq = len(trainSeq)
        # print trainSeq[1]
        self.obSeq = numpy.zeros(shape=(int(self.num_obSeq), ))
        # k = 0
        # while k < self.num_obSeq:
        #     self.obSeq[k] = trainSeq[k]
        #     # print trainSeq[k]
        #     k += 1
        self.obSeq=trainSeq

    def setObSeq(self, observationSeq):
        """ generated source for method setObSeq """

        self.currentSeq = observationSeq
        self.len_obSeq=len(observationSeq)
        self.alpha = numpy.zeros(shape=(int(self.len_obSeq), int(self.num_states)), dtype='float64')
        self.beta = numpy.zeros(shape=(int(self.len_obSeq), int(self.num_states)), dtype='float64')
        self.scaleFactor = numpy.zeros(shape=(int(self.len_obSeq),), dtype='float64')

    def reestimate(self):
        """ generated source for method reestimate """
        newTransition = numpy.zeros(shape=(int(self.num_states),int(self.num_states)), dtype='float64')
        newOutput = numpy.zeros(shape=(int(self.num_states),int(self.num_symbols)), dtype='float64')
        numerator = numpy.zeros(shape=(int(self.num_obSeq),))
        denominator = numpy.zeros(shape=(int(self.num_obSeq),), dtype='float64')
        cdef int sumP = 0
        cdef int i = 0
        cdef int j=0
        cdef int k=0
        cdef int denom=0
        while i < self.num_states:
            j=0
            while j < self.num_states:
                if j < i or j > i + self.delta:
                    newTransition[i][j] = 0
                else:
                    k=0
                    while k < self.num_obSeq:
                        numerator[k] = denominator[k] = 0
                        self.setObSeq(self.obSeq[k])
                        
                        sumP += self.computeAlpha()
                        self.computeBeta()
                        t=0
                        while t < self.len_obSeq - 1:
                            numerator[k] += self.alpha[t][i] * self.transition[i][j] * self.output[j][self.currentSeq[t + 1]] * self.beta[t + 1][j]
                            denominator[k] += self.alpha[t][i] * self.beta[t][i]
                            t += 1
                        k += 1
                    denom=0
                    k=0
                    while k < self.num_obSeq:
                        newTransition[i][j] += (1 / sumP) * numerator[k]
                        denom += (1 / sumP) * denominator[k]
                        k += 1
                    newTransition[i][j] /= denom
                    newTransition[i][j] += self.MIN_PROBABILITY
                j += 1
            i += 1
        sumP = 0
        i = 0
        
        while i < self.num_states:
            j=0
            while j < self.num_symbols:
                k=0
                while k < self.num_obSeq:
                    numerator[k] = denominator[k] = 0
                    self.setObSeq(self.obSeq[k])
                    # print self.obSeq[k]
                    sumP += self.computeAlpha()
                    self.computeBeta()
                    t=0
                    while t < self.len_obSeq - 1:
                        if self.currentSeq[t] == j:
                            numerator[k] += self.alpha[t][i] * self.beta[t][i]
                        denominator[k] += self.alpha[t][i] * self.beta[t][i]
                        t += 1
                    k += 1
                denom=0
                k=0
                while k < self.num_obSeq:
                    newOutput[i][j] += (1 / sumP) * numerator[k]
                    denom += (1 / sumP) * denominator[k]
                    k += 1
                newOutput[i][j] /= denom
                newOutput[i][j] += self.MIN_PROBABILITY
                j += 1
            i += 1
        self.transition = newTransition
        self.output = newOutput
        """
        for i in xrange(self.num_states):
            for j in xrange(self.num_symbols):
                for k in xrange(self.num_obSeq):
                    self.setObSeq(self.obSeq[k])
                    # print self.obSeq[k]
                    sumP += self.computeAlpha()
                    self.computeBeta()
                    alpha_times_beta = self.alpha[:,i] * self.beta[:,i]
                    numerator[k] = numpy.sum(alpha_times_beta[self.currentSeq == j])
                    denominator[k] = numpy.sum(alpha_times_beta)
                denom = numpy.sum(denominator)
                newOutput[i,j] = numpy.sum(numerator) / (sumP * denom) + self.MIN_PROBABILITY
        self.transition = newTransition
        self.output = newOutput
        """



    def train(self):
        """ generated source for method train """
        i = 0
        while i < 20:
            # print "Before"
            # print "OP", self.output,"PI", self.pi,"TR", self.transition
            self.reestimate()
            print "reestimating....." ,i
            i += 1




    def randomProb(self):
        """ generated source for method randomProb """


    def __init__(self, num_states, num_symbols):
        """ generated source for method __init___0 """
        self.num_states = num_states
        self.num_symbols = num_symbols
        self.transition = numpy.zeros(shape=(int(num_states), int(num_states)))
        self.output = numpy.zeros(shape=(int(num_states), int(num_symbols)))
        self.pi = numpy.zeros(shape=(int(num_states),))
        self.pi[0] = 1
        i = 1
        while i < num_states:
            self.pi[i] = 0
            i += 1
        """
        i = 0
        while i < self.num_states:
            j=0
            while j < self.num_states:
                if j < i or j > i + self.delta:
                    self.transition[i][j] = 0
                else:
                    self.transition[i][j] = random.random()
                j += 1
            j=0
            while j < self.num_symbols:
                self.output[i][j] = random.random()
                j += 1
            i += 1
        """
        self.output=[[ 0.36953436,0.19598372,0.08982479,0.26363117,0.26485188,0.87405232,0.3104701, 0.38166135,0.12646175,0.1246966, 0.14333271,0.42430546,0.33121599,0.98518463,0.91340862,0.30334883,0.70987023,0.76597139,0.44544497,0.8616658, 0.88460427,0.94827118,0.68413069,0.68006399,0.14682812,0.07880796,0.27239964,0.39178019,0.21225799,0.53520959,0.17185833,0.23372217,0.26197625,0.35665085,0.11342377,0.475398,0.53698304,0.50862632,0.71012491,0.74063726,0.45943932,0.4983742,0.3948246, 0.48789056,0.4161692, 0.16100604,0.45332451,0.00514567,0.15081786,0.63886889,0.41609237,0.99294587,0.62666633,0.49068639,0.48507395,0.32807081,0.1134539, 0.21477618,0.4683959, 0.77060636,0.54271318,0.21297853,0.35680518,0.90817007],[ 0.56750997,0.15244005,0.57872753,0.17904646,0.83457942,0.94508257, 0.54190004,0.01207668,0.61797715,0.57935944,0.54080001,0.72714057, 0.50938535,0.59957339,0.98452539,0.40230783,0.488466,0.5700868, 0.55679544,0.85365305,0.80603701,0.30369575,0.7379945, 0.4309972, 0.53364373,0.70728341,0.42538153,0.81275701,0.10110379,0.31078665, 0.20360748,0.04333151,0.53033057,0.54736895,0.13168782,0.0094116, 0.87848394,0.97880851,0.7997542, 0.4554397, 0.76747402,0.48458676, 0.66287739,0.27172192,0.48739451,0.24304105,0.86898709,0.4151162, 0.34180404,0.92666639,0.72899294,0.76194888,0.81717692,0.6688049, 0.53007975,0.7865035, 0.66509064,0.63136789,0.55432364,0.6821612, 0.12007275,0.47625354,0.09014107,0.87882442],[ 0.41411401,0.84730505,0.5314073, 0.42992594,0.20264881,0.62405617, 0.67016647,0.72496755,0.03482625,0.37809867,0.14168163,0.8557606, 0.9918291, 0.59131501,0.2762018, 0.05742589,0.25172495,0.01446863, 0.20896549,0.41350863,0.75484781,0.84355856,0.88981004,0.29459419, 0.29661281,0.12205803,0.72620028,0.21244065,0.03360511,0.73175435, 0.01610254,0.65124513,0.65391232,0.92256859,0.37134714,0.67954065, 0.19873429,0.26284617,0.67722258,0.71204053,0.88352127,0.90308002, 0.3377923, 0.59501051,0.52151985,0.5138592, 0.26118273,0.87601889, 0.36845128,0.71236852,0.91413352,0.46717065,0.02292141,0.5619901, 0.45836536,0.32425204,0.40693288,0.04078351,0.64945457,0.97943664, 0.63202215,0.70155233,0.50084791,0.81614766], [ 0.38053959,0.58521441,0.07066778,0.04638581,0.10468413,0.34626914, 0.41889552,0.8091748, 0.93604059,0.04059781,0.53413318,0.63002398, 0.46664192,0.7660146, 0.81854606,0.81047577,0.49999249,0.60572624, 0.6009544, 0.95918231,0.21148062,0.42079857,0.26195299,0.62601794, 0.98318005,0.29585371,0.51359912,0.96331065,0.24117602,0.45917561, 0.286697,0.22996295,0.93594394,0.46337351,0.12966658,0.35834149, 0.65270568,0.68700919,0.195269,0.06359174,0.64713036,0.28268207, 0.60993091,0.44978612,0.97099406,0.89933245,0.27079164,0.04509898, 0.07116063,0.28806396,0.79504335,0.52017146,0.88395056,0.53127009, 0.73536133,0.73507919,0.61389444,0.87332982,0.66874822,0.40690486, 0.21409875,0.33160526,0.46409337,0.79309219]]
        self.transition=[[ 0.43358209,  0.69263181,  0.76375569 , 0.        ],[ 0.        ,  0.16341849 , 0.20901959 , 0.25965116],[ 0.        ,  0.         , 0.46466536 , 0.65105618],[ 0.        ,  0.        ,  0.         , 0.78337595]]
        



    def save(self, modelName):
        """ generated source for method save """
        
        model = HMMModel()
        model.output=self.output
        model.pi=self.pi
        model.transition=self.transition
        with open(modelName, 'wb') as output:
            pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

    def initFromName(self, name):
        with open(name, 'rb') as input:
            newHMMM = pickle.load(input)
        # self.num_obSeq=newHMMM.num_obSeq
        self.output=newHMMM.output
        self.transition=newHMMM.transition
        self.pi=newHMMM.pi
        self.num_states=len(self.output)
        self.num_symbols=len(self.output[0])

