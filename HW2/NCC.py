import cv2
import numpy as np

def NCC():
    img_rgb = cv2.imread('./ncc_img.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('./ncc_template.jpg',0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.95
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    cv2.imshow('NCC',img_rgb)
    cv2.waitKey(1500)
    cv2.imshow('NCC',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



