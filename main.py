import cv2
from tracker import EuclideanDistTracker

tracker = EuclideanDistTracker()
cap = cv2.VideoCapture("Rodovia.mp4")

detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    print(f"Width: {width}, Height: {height}")
    
    roi = frame[100:340, 260:350]
    
    mask = detector.apply(roi)
    countors, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in countors:
        area = cv2.contourArea(cnt)
        if area > 1000:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detections.append([x, y, w, h])
            
        # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
    cv2.imshow("Roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()