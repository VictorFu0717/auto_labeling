from __future__ import print_function
import sys
import cv2
from random import randint
import time

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

# def createTrackerByName(trackerType):
#     # Create a tracker based on tracker name
#     if trackerType == trackerTypes[0]:
#         tracker = cv2.TrackerBoosting_create()
#     elif trackerType == trackerTypes[1]:
#         tracker = cv2.TrackerMIL_create()
#     elif trackerType == trackerTypes[2]:
#         tracker = cv2.TrackerKCF_create()
#     elif trackerType == trackerTypes[3]:
#         tracker = cv2.TrackerTLD_create()
#     elif trackerType == trackerTypes[4]:
#         tracker = cv2.TrackerMedianFlow_create()
#     elif trackerType == trackerTypes[5]:
#         tracker = cv2.TrackerGOTURN_create()
#     elif trackerType == trackerTypes[6]:
#         tracker = cv2.TrackerMOSSE_create()
#     elif trackerType == trackerTypes[7]:
#         tracker = cv2.TrackerCSRT_create()
#     else:
#         tracker = None
#         print('Incorrect tracker name')
#         print('Available trackers are:')
#         for t in trackerTypes:
#             print(t)
#     return tracker



if __name__ == '__main__':

    # print("Default tracking algoritm is CSRT \n"
    #     "Available tracking algorithms are:\n")
    # for t in trackerTypes:
    #     print(t)

    # 準備三個空資料夾, datasets & images & labels

    trackerType = "CSRT"

    # 影片路徑
    videoPath = "C:/Users/victor/PycharmProjects/yolov5_old/Auto_labeling/541.MOV"

    cap = cv2.VideoCapture(videoPath)
    # 設定擷取影像的尺寸大小
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

    # init
    bboxes = []
    colors = []
    class_list = []

    # 起始檔名 & 計數
    count = 1

    # 每幾楨存一次
    n_save = 3


    while True:
        success, frame = cap.read()
        cv2.imshow("Frame", frame)
        # time.sleep(0.1)
        key = cv2.waitKey(1) & 0xFF
        (H, W) = frame.shape[:2]


        # 按下s後影片暫停
        if key == ord("s"):
            start_time = time.time()
            while True:
                # 輸入class id 數字
                class_id = input('input the class_id : ')
                class_list.append(eval(class_id))
                # 框出要追蹤的物件
                bbox = cv2.selectROI('Frame', frame)
                bboxes.append(bbox)
                colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
                print("Press q to quit selecting boxes and start tracking")
                print("Press any other key to select next object")
                k = cv2.waitKey(0) & 0xFF
                if (k == 113):  # q is pressed
                    break
            print('Selected bounding boxes {}'.format(bboxes))

            # Create MultiTracker object
            multiTracker = cv2.legacy.MultiTracker_create()

            # Initialize MultiTracker
            for bbox in bboxes:
                tracker = cv2.legacy.TrackerCSRT_create()
                multiTracker.add(tracker, frame, bbox)

            # Process video and track objects
            while cap.isOpened():
                count += 1
                success, frame = cap.read()

                if not success:
                    break
                # 存下訓練用圖片, 沒有BBOX
                if count % n_save == 0:
                    cv2.imwrite(f"C:/Users/victor/PycharmProjects/yolov5_old/Auto_labeling/images/{count}.jpg", frame)
                # get updated location of objects in subsequent frames
                success, boxes = multiTracker.update(frame)

                fps = round(count / (time.time() - start_time))
                info = [
                    ("Tracker", trackerType),
                    ("Success", "Yes" if success else "No"),
                    ("FPS", fps),
                ]
                # loop over the info tuples and draw them on our frame
                # for (i, (k, v)) in enumerate(info):
                #     text = "{}: {}".format(k, v)
                #     cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # print(class_list)


                # 座標處理
                for i, newbox in enumerate(boxes):
                    x = int(newbox[0])
                    y = int(newbox[1])
                    w = int(newbox[2])
                    h = int(newbox[3])
                    n_x = (x + (w / 2)) / W
                    n_y = (y + (h / 2)) / H
                    n_w = w / W
                    n_h = h / H
                    cv2.rectangle(frame, (x,y), (x+w, y+h), colors[i], 2, 1)
                    # 存下訓練用座標檔 yolov5格式
                    if count % n_save == 0:
                        with open(f"C:/Users/victor/PycharmProjects/yolov5_old/Auto_labeling/labels/{count}.txt", "a",
                                  encoding="utf-8") as f:
                            f.write(f"{class_list[i]} {n_x} {n_y} {n_w} {n_h}\n")

                    # show frame
                cv2.imshow('Frame', frame)
                # 存下有畫BBOX的圖片,不用於訓練, 後續篩選用
                if count % n_save == 0:
                    cv2.imwrite(f"C:/Users/victor/PycharmProjects/yolov5_old/Auto_labeling/dataset/{count}.jpg", frame)


                    # quit on ESC button
                if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                    break


        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
