import cv2 
import numpy as np
import RPi.GPIO as pi
import time

def pi_relay_control():
    pin1 = 16
    pin2 = 18
    pi.setmode(pi.BOARD)
    pi.setup(pin1,pi.OUT)
    pi.setup(pin2,pi.OUT)
    pi.output(pin1,False)
    pi.output(pin2,False)
    try:
        while True:
            for x in range(5):
                pi.output(pin1,True)
                time.sleep(0.5)
                pi.output(pin1,False)
                pi.output(pin2,True)
                time.sleep(0.5)
                pi.output(pin2,False)
            pi.output(pin1,True)
            pi.output(pin2,True)
            for x in range (4):
                pi.output(pin1,True)
                time.sleep(0.05)
                pi.output(pin1,False)
                time.sleep(0.05)
            pi.output(pin1,True)
            for x in range (4):
                pi.output(pin2,True)
                time.sleep(0.05)
                pi.output(pin2,False)
                time.sleep(0.05)
            pi.output(pin2,True)
    except KeyboardInterrupt:
        pi.cleanup()


#fun to capture histogram

def capture_histogram(path_of_sample):
    color = cv2.imread(path_of_sample)
    
    color_hsv = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)

    object_hist = cv2.calcHist([color_hsv],[0,1],None,[180,256],[0,180,0,256])# image,channels,no mask,size of hist,chan val
    
    cv2.normalize(object_hist,object_hist,0,255,cv2.NORM_MINMAX) 
    return object_hist
    
# locate the object in a frame

def locate_object(frame,object_hist):
    hsv_frame =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #apply back projection to image using object_hist as
    # the model histogram
    object_segment = cv2.calcBackProject([hsv_frame],[0,1],object_hist, [0,180,0,256],1)

    #find contours
    img,contours, _ = cv2.findContours(object_segment,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    

    flag = None
    max_area = 0

    #find the contour with greatest area
    for(i,c) in enumerate(contours):
        area = cv2.contourArea(c)
        if area >max_area:
            max_area = area
            flag = i
    #get the rectangle
    if flag is not None and max_area>1000:
        cnt = contours[flag]
        coords = cv2.boundingRect(cnt)
        return coords
    return None

hist = capture_histogram('color_sample.png')

cap= cv2.VideoCapture('sam_vid.mp4')

while True:

    _,frame = cap.read()

    coords = locate_object(frame,hist) #back projects the hist and finds similar frame
    
    if coords:
        #unpack coord
        x,y,w,h = coords
        print(x,y,w,h)
        
        #draw the bounding box

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        pi_relay_control()
    
    cv2.imshow("Object Detection Using Color",frame)


    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

