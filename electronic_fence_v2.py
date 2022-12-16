# 電子圍籬 ＆ 遺留物偵測
import cv2
import numpy as np

# 開啟網路攝影機
cap = cv2.VideoCapture(0)

# 設定影像尺寸
width = 640
height = 480

# 紀錄停留時間
retentateTimerMap = np.zeros((height,width,1), dtype = "uint8")
retentateTimerMapTmp = np.zeros((height,width,1), dtype = "uint8")

# 設定擷取影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 計算畫面面積
area = width * height


# 初始化平均影像
ret, frame = cap.read()
avg = cv2.blur(frame, (4, 4))
# copy = np.zeros((height,width,1), dtype="uint8")
copy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
avg_float = np.float32(avg)

while(cap.isOpened()):
  # 讀取一幅影格
  ret, frame = cap.read()

  # 若讀取至影片結尾，則跳出
  if ret == False:
    break

  # 模糊處理
  blur = cv2.blur(frame, (4, 4))

  # 計算目前影格與平均影像的差異值
  diff = cv2.absdiff(avg, blur)

  # 將圖片轉為灰階
  gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

  # 篩選出變動程度大於門檻值的區域
  ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

  # 使用型態轉換函數去除雜訊
  kernel = np.ones((5, 5), np.uint8)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

  # 產生等高線
  cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # print(cnts)
  retentateTimerMapTmp = np.zeros((height, width, 1), dtype="uint8")
  if len(cnts) != 0:
    for c in cnts:
      # 忽略太小的區域
      if cv2.contourArea(c) > 800:
        # 計算等高線的外框範圍
        (x, y, w, h) = cv2.boundingRect(c)
        retentateTimerMapTmp[y:y + h,x:x + w] = 255
        retentateTimerMap[y:y + h,x:x + w] += 1 # 計次
        # 超過100次判定為遺留物
        if (retentateTimerMap >= 100).any():
          # 偵測到遺留物，可以自己加上處理的程式碼在這裡...

          print("alarm")
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

  retentateTimerMap[retentateTimerMapTmp != 255] //= 2

  # 做遮罩，將遺留物的部分做成遮罩，停止更新，以免遺留物和背景融為一體；若未出現遺留物，初始化遮罩，去除遮罩上的雜訊
  mask = np.ones((height,width,1), dtype = "uint8")*255
  if (retentateTimerMapTmp == 255).any():
    cv2.drawContours(mask, cnts, -1, 0, -1)
  cv2.imshow('mask',mask)

  # 顯示偵測結果影像
  cv2.imshow('frame', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  # 更新平均影像
  cv2.accumulateWeighted(blur, avg_float, 0.005, mask = mask)
  avg = cv2.convertScaleAbs(avg_float)
  cv2.imshow('avg',avg)
  cv2.imshow('retentateTimerMap', retentateTimerMap)
  cv2.imshow('retentateTimerMapTmp', retentateTimerMapTmp)

cap.release()
cv2.destroyAllWindows()
