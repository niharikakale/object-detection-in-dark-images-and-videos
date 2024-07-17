import cv2


from ultralytics import YOLO
import supervision as sv
import numpy as np

def main():

    cap=cv2.VideoCapture(r"C:\Users\Gowthami\Contacts\Desktop\stock-footage-night-traffic-jam-timelapse-bangkok-thailand-rush-hour-in-downtown-motion-of-car-tail-lights.webm")

    model=YOLO("yolov8l.pt")

    box_annotator=sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,text_scale=1
    )
    while True:
        ret,frame=cap.read()
        #frame =cv2.resize(frame,(1020,500))
        result=model.predict(frame,agnostic_nms=True)[0]
        detections=sv.Detections.from_yolov8(result)
        #detections=detections[detections.class_id==0]
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _,_,confidence, class_id, _
            in detections
        ]


        frame=box_annotator.annotate(scene=frame,
                                     detections=detections,
                                     labels=labels)

        cv2.imshow("frame",frame)
        cv2.waitKey(10)


if  __name__ == "__main__":
    main()


