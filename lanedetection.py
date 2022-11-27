import cv2
import numpy as np
 
def canny(img):
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 50, 150)
    return canny

cap = cv2.VideoCapture("test1.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)

    cv2.imshow("canny_image",canny_image)
#
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

