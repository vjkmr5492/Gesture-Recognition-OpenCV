import cv2
import cv2.cv as cv
import numpy as np
import os, pickle
import time, pprint
from GestureFeatureExtractor import GestureFeatureExtractor as GFE
from scipy.cluster.vq import kmeans, vq, whiten
from DiscreteHMM import DiscreteHMM
from HiddenMarkov import HiddenMarkov
from Codebook import Codebook
from Points import Points

gname="UL"
# tname="UL"
started=False
ar=0

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


def on_mouse(event, x, y, flags, params):
    global started
    global ar
    if event == cv.CV_EVENT_LBUTTONDOWN:
        # print 'Start Mouse Position: '+str(x)+', '+str(y)
        if started==False:
            started=True
        else:
            started=False
        sbox = [x, y]
        boxes.append(sbox)
    elif event == cv.CV_EVENT_MOUSEMOVE:
        ar+=1
        if started==True and ar%5==0:
            ebox = [x, y]
            boxes.append(ebox)
            # print ebox
            
    

def directory(name):
    if not os.path.exists(name):
        os.makedirs(name)

def gen_cb():
    print "Generating Codebook..."
    dirs = [name for name in os.listdir(".") if os.path.isdir(os.path.join(".", name)) and name!='data']
    for diri in dirs:
        vector=[]
        print "Working for "+diri
        raws = [name for name in os.listdir("./"+diri) if name.endswith(".raw")]
        for name in raws:
            print "Processing file "+name
            rl =  load_object("./"+diri+"/"+name)
            ob = GFE(rl)
            for i in range(len(rl)-1):
                vector.append(ob.extractedFeature[i].getNFeatureVector())
    X=np.vstack(vector).squeeze()
    print X
    directory("data")
    c=Codebook()
    c.genCB(X)
    c.saveToFile()


def quantize(rawdata):
    ob = GFE(rawdata)
    qv=[]
    for i in range(len(rawdata)-1):
        qv.append(ob.extractedFeature[i].getFeatureVector())
    X=np.vstack(qv).squeeze()
    cbk = [name for name in os.listdir("./data") if name.endswith(".cbk")]
    if len(cbk)==1:
        init=load_object("./data/data.cbk")
    else:
        init=64
    # whiten(X)
    codebook, distortion = kmeans(X, init, 1)
    code, dist = vq(X, codebook)
    # print code
    return code


boxes=[]

# cap = cv2.VideoCapture(0)
# _,img = cap.read()
img=np.zeros(shape=(480,640,3))
while(1):
    cv2.namedWindow('real image')
    cv.SetMouseCallback('real image', on_mouse, 0)
    for point in range(len(boxes)-1):
        cv2.line(img,(boxes[point][0], boxes[point][1]),(boxes[point+1][0], boxes[point+1][1]),[255,255,0],2)
    cv2.imshow('real image', img)
    img=np.zeros(shape=img.shape)
    
    
    c=cv2.waitKey(1)
    if c==27:
        
        break
    elif c==ord('s'):
        directory(gname)
        fname=int(time.time())
        save_object(boxes, "./"+gname+"/"+str(fname)+".raw")
    elif c==ord('g'):
        gen_cb()
    elif c==ord('r'):
        boxes=[]
    elif c==ord('q'):
        cb=Codebook()
        cb.load()
        print len(cb.centroids)
        ob = GFE(boxes)
        qv=[]
        for i in range(len(boxes)-1):
            qv.append(  Points(ob.extractedFeature[i].getNFeatureVector())  )
        X=np.vstack(qv).squeeze()
        print cb.quantize(qv)

        
    elif c==ord('t'):
        cb=Codebook()
        cb.load()
        dirs = [name for name in os.listdir(".") if os.path.isdir(os.path.join(".", name)) and name!='data']
        tmp=[]
        for tname in dirs:
            
            raws = [name for name in os.listdir("./"+tname) if name.endswith(".raw")]
            for name in raws:
                rl =  load_object("./"+tname+"/"+name)
                ob = GFE(rl)
                fqv=[]
                for i in range(len(rl) - 1):
                    fqv.append(  Points(ob.extractedFeature[i].getNFeatureVector())  )
                tmp.append(cb.quantize(fqv))
            mkv=HiddenMarkov(4, 64)
            print "Starting training for ", tname
            mkv.setTrainSeq(np.array(tmp))
            print "training", tname
            mkv.train()
            print "Saving", tname
            mkv.save("./"+tname+"/"+tname+".hmm")
        print "Training done"
    elif c==ord('v'):
        """
        line=HiddenMarkov(4, 64)
        line.initFromName("./line/line.hmm")

        L=HiddenMarkov(4, 64)
        L.initFromName("./L/L.hmm")
        UL=HiddenMarkov(4, 64)
        UL.initFromName("./UL/UL.hmm")
        """
        ml=[]
        # boxi=[40,40,41,4,4,5,5,7,6,1,1,26,1,33,36,36,36,38,38,38]

        cb=Codebook()
        cb.load()
        ob = GFE(boxes)
        qv=[]
        for i in range(len(boxes)-1):
            qv.append(  Points(ob.extractedFeature[i].getNFeatureVector())  )
        X=np.vstack(qv).squeeze()
        qt = cb.quantize(qv)

        dirs = [name for name in os.listdir(".") if os.path.isdir(os.path.join(".", name)) and name!='data']
        for md in dirs:
            x=HiddenMarkov(4,64)
            x.initFromName("./"+md+"/"+md+".hmm")
            ml.append(x)
            x=None
        lk=[]
        for i in ml:
            lk.append(i.viterbi(qt))
        """
        print line.viterbi(quantize(boxes))

        print L.viterbi(quantize(boxes))
        print UL.viterbi(quantize(boxes))
        """
        maxi=-50000
        for i in range(len(lk)):
            if lk[i] > maxi:
                maxi = lk[i]
                maxIndex = i
        print dirs[maxIndex]
        for i in range(len(ml)):
            print lk[i], dirs[i]
    elif c==ord('h'):
        hname="UL"
        td_line=[[59,59,58,58,58,55,55,54,55,55,55,53,53,53,53,53,53,60,60],[59,57,58,57,57,59,59,54,54,54,57,57,59,57,57,57,57,60],[50,39,50,50,50,50,50,50,50,52,52,52,52,53,53,53,53,57,57,57,57,54,55,55,55,58,58,57,57,57,57,57,57,57,57,57,57,57,60],[59,58,58,55,56,59,54,54,56,55,55,55,53,53,60],[59,49,57,57,57,57,57,57,57,57,57,57,56,57,57,57,56,57,57,56,57,57,57,57,57,53,53,53,53,53,52,52,60]]
        td_l=[[10,32,12,2,2,2,3,3,49,49,56,56,56,37,37,22,21,24,27,28,29,29,29,60],[10,10,0,0,0,2,2,2,2,2,3,3,3,16,16,23,23,22,21,20,24,24,24,25,25,25,25,28,28,60],[10,10,2,2,2,2,2,2,2,2,3,3,15,34,35,35,35,38,37,37,36,36,22,22,21,20,24,24,28,27,28,30,29,29,29,60,60],[10,10,2,2,2,2,3,3,3,16,16,16,23,23,23,22,22,20,24,24,25,28,28,27,31,31],[10,11,12,2,2,2,2,2,2,2,3,3,3,16,16,16,23,23,23,22,22,21,21,20,24,24,25,27,27,28,28,29,60,60]]
        td_ul=[[47,41,4,4,4,4,5,5,5,7,7,7,6,1,1,3,26,26,33,33,36,36,36,37,38,38,38,39,39,60,60],[41,41,41,41,4,4,4,4,5,5,5,5,7,7,6,1,1,1,1,1,26,33,33,36,36,36,36,37,37,38,39,39],[40,41,41,4,4,4,4,4,4,4,5,5,5,7,6,1,1,1,1,26,26,24,24,24,33,36,36,36,36,36,36,37,60,60],[40,42,42,42,42,42,41,41,41,5,7,6,6,1,1,1,1,24,1,33,36,36,36,38,38,38]]
        mkv=HiddenMarkov(4, 64)
        mkv.setTrainSeq(np.array(td_ul))
        mkv.train()
        mkv.save("./"+hname+"/"+hname+".hmm")





        
cv2.destroyAllWindows()
    