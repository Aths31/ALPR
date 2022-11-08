import cv2
import numpy as np
import pytesseract

frameWidth = 640    #Frame Width
franeHeight = 480   # Frame Height

plateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
minArea = 500

cap =cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,franeHeight)
cap.set(10,150)
count = 0

while True:
    success , img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            wT,hT,cT=img.shape
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img,"NumberPlate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("ROI",imgRoi)
            cv2.imshow("Result",img)
        if cv2.waitKey(1) & 0xFF ==ord('s'):
            cv2.imwrite(".\IMAGES"+str(count)+".jpg",imgRoi)
            cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
            cv2.putText(img,"Scan Saved",(15,265),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
            cv2.imshow("Result",img)
            a,b=(int(0.02*wT),int(0.02*hT))
            plate=img[y+a:y+h-a,x+b:x+w-b,:]
            #make the img more darker to identify LPR
            kernel=np.ones((1,1),np.uint8)
            plate=cv2.dilate(plate,kernel,iterations=1)
            plate=cv2.erode(plate,kernel,iterations=1)
            plate_gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
            read=pytesseract.image_to_string(plate)
            print(read)
            cv2.waitKey(500)
            count+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

