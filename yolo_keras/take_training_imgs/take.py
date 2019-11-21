import cv2
import argparse

parser = argparse.ArgumentParser("description='--num:how many imgs to take, --category:class name now take'")
parser.add_argument("--num", type=int,default=200)
parser.add_argument("--category", type=str)
args = parser.parse_args()

cap = cv2.VideoCapture(0)
cap.set(3,416)
cap.set(4,416)
i = 0
flag = False
while(i < args.num):
    _ , frame = cap.read()
    cv2.imshow("capture img",frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
    if k == ord('s'):
        flag = True
        print("start take")
    if k == ord('i'):
        flag = False
        print("stop take")
    if flag is True:
        cv2.imwrite("./images/%s_%d.jpg"%(args.category,i),frame)
        i = i + 1
        print("now take %d"%i)
