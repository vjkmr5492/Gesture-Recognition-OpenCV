import cv2
import cv2.cv as cv
import numpy as np
import os, pickle
import time, pprint
from GestureFeatureExtractor import GestureFeatureExtractor as GFE
from scipy.cluster.vq import kmeans, vq, whiten
from DiscreteHMM import DiscreteHMM
from mycython import HiddenMarkov
from Codebook import Codebook
from Points import Points
from Centroid import Centroid

gname="Circle"
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
        if started==True:
            ebox = [x, y]
            boxes.append(ebox)
            # print ebox
            
    

def directory(name):
    if not os.path.exists(name):
        os.makedirs(name)

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
        # boxes=[[355,137],[354,137],[353,137],[352,137],[351,137],[350,137],[349,137],[348,137],[347,137],[345,137],[344,137],[342,137],[338,137],[336,137],[332,137],[327,137],[322,138],[320,138],[317,138],[315,138],[313,138],[311,137],[309,137],[308,137],[305,137],[304,137],[302,137],[301,137],[300,137],[298,136],[295,136],[294,136],[292,136],[288,136],[286,136],[285,136],[282,134],[279,134],[278,134],[276,134],[275,135],[274,135],[272,135],[269,135],[268,135],[266,135],[265,135],[262,135],[260,136],[259,136],[257,136],[253,136],[251,136],[249,136],[246,136],[241,136],[239,136],[237,136],[235,137],[234,137],[232,137],[228,137],[226,137],[221,137],[219,137],[218,137],[217,137],[215,137],[213,137],[209,137],[205,137],[201,137],[198,137],[197,137],[191,137],[188,137],[186,137],[184,137],[182,137],[181,137],[180,137],[178,137],[177,137],[175,137],[173,137],[172,137],[171,137],[168,137],[164,137],[161,137],[158,137],[155,139],[152,139],[147,140],[145,140],[143,142],[140,143],[138,143],[136,143],[136,144],[133,144],[133,145],[133,146],[130,146],[129,146],[128,146],[126,146],[125,145],[124,145],[123,144],[122,147],[121,147],[120,148],[119,148]]
        # boxes=[[357,130],[356,130],[355,130],[354,130],[353,130],[352,130],[351,130],[350,129],[348,128],[344,128],[341,128],[334,128],[328,128],[319,127],[312,127],[303,127],[294,129],[291,129],[285,129],[281,130],[276,130],[273,130],[271,130],[267,130],[262,128],[259,128],[254,128],[251,127],[246,127],[244,127],[240,127],[239,127],[236,127],[233,126],[229,125],[226,125],[224,125],[220,125],[217,125],[214,125],[209,125],[202,119],[198,119],[195,117],[193,117],[191,117],[187,117],[185,117],[183,117],[181,117],[179,117],[174,117],[172,117],[169,117],[165,117],[162,117],[159,117],[155,118],[150,118],[142,118],[137,118],[133,118],[129,118],[127,118],[124,118],[122,118],[118,118],[115,118],[111,118],[108,120],[104,120],[101,119],[93,118],[90,117],[84,116],[82,116],[81,116],[80,116],[79,116],[79,115]]
        # boxes=[[156,48],[156,50],[156,52],[156,54],[156,56],[156,58],[157,65],[159,68],[160,77],[160,79],[160,84],[160,88],[160,93],[160,97],[160,101],[160,105],[160,108],[160,118],[162,131],[163,135],[163,137],[163,141],[163,145],[163,147],[163,149],[163,154],[163,157],[163,159],[163,163],[164,166],[164,171],[164,177],[167,183],[166,196],[166,199],[166,202],[166,206],[166,208],[166,210],[165,212],[165,213],[165,215],[165,216],[165,217],[165,218],[166,218],[167,217],[168,217],[170,216],[176,216],[177,216],[180,216],[186,216],[190,215],[198,215],[203,215],[211,213],[218,213],[228,213],[235,213],[244,213],[247,213],[252,213],[255,212],[259,212],[265,212],[266,212],[269,212],[270,212],[272,212],[273,212],[275,212],[278,212],[282,212],[283,211],[285,211],[287,211],[288,210],[289,210]]
        cb=Codebook()
        cb.load()
        # print "CLEN" , len(cb.centroids)
        ob = GFE(boxes)
        qv=[]
        for i in range(len(boxes)-1):
            qv.append(  Points(ob.extractedFeature[i].getNFeatureVector())  )
        X=np.vstack(qv).squeeze()
        # X=qv
        print cb.quantize(X)

        
    elif c==ord('t'):
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
    elif c==ord('v'):

        ml=[]
        
        # boxi=[40,40,41,4,4,5,5,7,6,1,1,26,1,33,36,36,36,38,38,38]
        cb=Codebook()
        cb.load()
        ob = GFE(boxes)
        qv=[]
        for i in range(len(boxes)-1):
            qv.append(  Points(ob.extractedFeature[i].getNFeatureVector())  )
        X=np.vstack(qv).squeeze()
        qt = cb.quantize(X)
        print qt
        
        # qt=[6,6,7,7,7,7,2,2,3,3,3,3,3,5,5,5,20,20,20,21,21,21,22,22,22,22,22,19,19,19,19,19,19,19,19,19,19,19,19,18,18,18,8,8,9,9,9,11,11,14,14,14,14,14,29,29,29,29,30,30,27,25,24,15]
        dirs = [name for name in os.listdir(".") if os.path.isdir(os.path.join(".", name)) and name!='data' and name!='build']
        for md in dirs:
            x=HiddenMarkov(4,64)
            x.initFromName("./"+md+"/"+md+".hmm")
            ml.append(x)
            print md, x.viterbi(np.array(qt))
            x=None
        


    elif c==ord('f'):
        ob = GFE(boxes)
        fqv=[]
        for i in range(len(boxes) - 1):
            fqv.append(  ob.extractedFeature[i].getNFeatureVector()  )
        print fqv



                





        
cv2.destroyAllWindows()
    