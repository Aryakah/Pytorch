import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model(["image1.jpg", "image2.jpg"])

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk


pathm = "image1.jpg"
results2 = model(pathm)
for r in results2:
    print(r.masks) 




pathb = "file.mp4"
cap = cv2.VideoCapture(0)


while cap.isOpened():
    
    success, frame = cap.read()

    if success:
     
        results = model(frame)

       
        annotated_frame = results[0].plot()

        
        cv2.imshow("YOLO Inference", annotated_frame)

       
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
      
        break


cap.release()
cv2.destroyAllWindows()