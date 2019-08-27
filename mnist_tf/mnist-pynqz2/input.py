import cv2;
import numpy as np

frame = np.zeros((500, 500, 3), np.float32) 

last_measurement = current_measurement = np.array((0, 0), np.float32)

def OnMouseMove(event, x, y, flag, userdata):
    global frame, current_measurement, last_measurement
    if event == cv2.EVENT_LBUTTONDOWN:
        current_measurement = np.array([[np.float32(x)], [np.float32(y)]]) # 当前测量

    if event == cv2.EVENT_MOUSEMOVE and flag == cv2.EVENT_FLAG_LBUTTON: 
        last_measurement = current_measurement # 把当前测量存储为上一次测量
        current_measurement = np.array([[np.float32(x)], [np.float32(y)]]) # 当前测量
        lmx, lmy = last_measurement[0], last_measurement[1] # 上一次测量坐标
        cmx, cmy = current_measurement[0], current_measurement[1] # 当前测量坐标
        cv2.line(frame, (lmx, lmy), (cmx, cmy), (255, 255, 255), thickness = 60) #输入数字    
# 窗口初始化
cv2.namedWindow("Input Number:")
#opencv采用setMouseCallback函数处理鼠标事件，具体事件必须由回调（事件）函数的第一个参数来处理，该参数确定触发事件的类型（点击、移动等）
cv2.setMouseCallback("Input Number:", OnMouseMove)
key = 0
while key != ord('q'):
    cv2.imshow("Input Number:", frame)
    key = cv2.waitKey(1) & 0xFF
res = cv2.resize(frame,(28,28))
res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
res = res/255.0
#print(res)
cv2.imwrite('tmp.bmp',res)
print('number image has been stored and named "tmp.bmp"')
cv2.destroyAllWindows()
