import cv2
import numpy as np
import glob, os, sys


def read_image(path):
    img = cv2.imread(path)
    return img

def image_list():
    image_data = []
    global flname
    flname = []
    path = "C:/tenjumh/Semantic-Segmentation/CamVid/train/*.tif"
    files = sorted(glob.glob(path))

    for fl in files:
        filename = os.path.basename(fl)

        img = read_image(fl)
        image_data.append(img)
        flname.append(filename)

    return image_data

def boundingbox():
    X = []
    for im in image_list():
        img = cv2.fastNlMeansDenoisingColored(im, None, 10, 10, 7, 21)  # 노이즈 제거
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        dila = cv2.dilate(gray, kernel, iterations=2)
        erod = cv2.erode(dila, kernel, iterations=2)
        blur = cv2.GaussianBlur(erod, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        rect = []
        rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            xw = x + w
            yh = y + h
            rect.append([x, y, w, h, xw, yh])
        x = rect[0][0]
        y = rect[0][1]
        xw = rect[0][4]
        yh = rect[0][5]
        for i in range(1, len(rect)):
            if x > rect[i][0]:
                x = rect[i][0]
            if y > rect[i][1]:
                y = rect[i][1]
            if xw < rect[i][4]:
                xw = rect[i][4]
            if yh < rect[i][5]:
                yh = rect[i][5]
            w = xw - x
            h = yh - y
        rects.append((x, y, w, h, xw, yh))
        for r in rects:
            x, y, w, h, xw, yh = r
        num = gray[y:yh, x:xw]
        num = 255 - num
        ww = round((w if w > h else h) * 1.85)
        spc = np.zeros((ww, ww))
        wy = (ww - h) // 2
        wx = (ww - w) // 2
        spc[wy:wy+h, wx:wx+w] = num
        num = cv2.resize(spc, (128, 128))
        #cv2.imwrite(str(con)+"-num.PNG", num)
        X.append(num)
        #print(y, yh, yh-y, x, xw, xw-x)
        #print(wy, wy+h, h, wx, wx+w, w)
        #cv2.waitKey(0)
        #red = (0, 0, 255)
        #cv2.rectangle(img, (x, y), (xw, yh), red, 2)
        #cv2.imshow("img", img)
        #cv2.rectangle(img, (x, y), (xw, yh), red, 2)
        #print(x, y, xw, yh)
        #red = (0, 0, 255)
        #cv2.rectangle(im, (x, y), (xw, yh), red, 2)
        #cv2.imshow("im", im)
        #cv2.waitKey(0)

    return X

def create_image():
    for int, img_resize in enumerate(boundingbox()):
        filename = os.path.splitext(flname[int])
        filename = filename[0]
        img = img_resize
        cv2.imwrite("C:/tenjumh/Semantic-Segmentation/CamVid/train/" + filename +".PNG", img)

create_image()