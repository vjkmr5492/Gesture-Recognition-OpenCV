import math
import numpy
import random
from HMMModel import HMMModel
import pickle

class HiddenMarkov(object):
       
    MIN_PROBABILITY = 0.00000000001
  
   
   
    delta = 2
    obSeq=[]
   
  
   

    # 
    #    * viterbi algorithm used to get best state sequence and probability<br>
    #    * calls: none<br>
    #    * called by: volume
    #    * 
    #    * @param testSeq
    #    *            test sequence
    #    * @return probability
    #    
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
        setObSeq(testSeq)
        temp = computeAlpha()
        return temp

    def computeAlpha(self):
        """ generated source for method computeAlpha """
        probability = 0
        t = 0
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
        i = 0
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
        self.alpha = numpy.zeros(shape=(int(self.len_obSeq), int(self.num_states)))
        self.beta = numpy.zeros(shape=(int(self.len_obSeq), int(self.num_states)))
        self.scaleFactor = numpy.zeros(shape=(int(self.len_obSeq),))

    def reestimate(self):
        """ generated source for method reestimate """
        newTransition = numpy.zeros(shape=(int(self.num_states),int(self.num_states)))
        newOutput = numpy.zeros(shape=(int(self.num_states),int(self.num_symbols)))
        numerator = numpy.zeros(shape=(int(self.num_obSeq),))
        denominator = numpy.zeros(shape=(int(self.num_obSeq),))
        sumP = 0
        i = 0
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

    def train(self):
        """ generated source for method train """
        i = 0
        while i < 20:
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

