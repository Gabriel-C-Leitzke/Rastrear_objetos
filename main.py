import cv2

cap = cv2.VideoCapture("Rodovia.mp4")

detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20)

while True:
    ret, frame = cap.read()
    
    roi = frame[100:340, 250:360]
    
    mask = detector.apply(roi)
    countors, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in countors:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detections.append([x, y, w, h])
            
    cv2.imshow("Roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()