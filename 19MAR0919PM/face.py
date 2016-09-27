import cv2, time
import cv2.cv as cv

cap = cv2.VideoCapture(0)
time.sleep(1)
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
cv.NamedWindow('w2', cv.CV_WINDOW_AUTOSIZE)
def detect(image):
    faces = cascade.detectMultiScale(image)
    for _face in faces:
        cv2.rectangle(image, (_face[0], _face[1]), (_face[0]+_face[2], _face[1]+_face[3]), (255,255,255))
        roi=image[_face[0]:_face[1], _face[0]+_face[2]: _face[1]+_face[3]]
        
        
        print _face[0],":",_face[1],",", _face[0]+_face[2],":", _face[1]+_face[3]
        return _face


while True:
    ret, image = cap.read()
    faces = cascade.detectMultiScale(image)
    if len(faces)>0:
        _face=faces[0]
        cv2.rectangle(image, (_face[0], _face[1]), (_face[0]+_face[2], _face[1]+_face[3]), (255,255,255))
        
        cv2.imshow("w1", image)
        roi=image[_face[0]:_face[1], _face[0]+_face[2]: _face[1]+_face[3]]
        roi=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mn=cv2.mean(roi)
        print mn
    c = cv.WaitKey(5)
    if c == 27 : 
        break