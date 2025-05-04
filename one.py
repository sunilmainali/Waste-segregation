
import cv2
from ultralytics import YOLO


model = YOLO("best.pt")

# Load the class names (COCO dataset in this example)
class_names = model.names  # This will give you the class names directly from the model
# Initialize webcam
cap = cv2.VideoCapture(0)


# Loop to read frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the frame
    results = model(frame)[0]

    # Extract bounding boxes, class ids, and scores
    boxes = results.boxes.xyxy.numpy()  # Bounding box coordinates (x1, y1, x2, y2)
    confs = results.boxes.conf.numpy()  # Confidence scores
    classes = results.boxes.cls.numpy()  # Class IDs

    
    # Annotate bounding boxes and labels
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        conf = confs[i]
        class_id = int(classes[i])

        # Get the class name
        class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

        # Print Class ID and Class Name
        print(f"Class ID: {class_id}, Class Name: {class_name}, Confidence: {conf:.2f}")

        label = f"{class_name} Conf {conf:.2f}"

        # Draw the bounding box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the annotated frame
    cv2.imshow('YOLO Inference', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

