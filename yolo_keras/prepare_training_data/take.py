import cv2
import time
import argparse

parser = argparse.ArgumentParser("description='--num:how many imgs to take, --category:class name now take'")
parser.add_argument("--num", type=int,default=200)
parser.add_argument("--dir", type=str,default='images2')
args = parser.parse_args()

cap = cv2.VideoCapture(1)
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
        flag = not flag
    if flag:
        time.sleep(1)
        ticks = time.time()
        cv2.imwrite("./"+args.dir+"/"+str(ticks)+".jpg",frame)
        i = i + 1
        print("now take %d"%i)
