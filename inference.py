import tensorflow as tf
import numpy as np
import cv2

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="your_model.tflite")
interpreter.allocate_tensors()

# Input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# OpenCV video capture (you can modify this to use your camera)
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (you may need to adjust preprocessing based on your model)
    frame = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    frame = frame / 255.0  # Normalize to [0, 1]
    frame = frame.astype(np.float32)  # Convert to FLOAT32
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], frame)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-process the output (you may need to adjust this based on your model)
    # For example, draw bounding boxes on the frame
    for detection in output_data[0]:
        score = detection[2]
        if score > 0.5:  # Adjust the confidence threshold as needed
            class_id = int(detection[1])
            box = detection[0]
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
            xmin = int(xmin * frame.shape[2])
            xmax = int(xmax * frame.shape[2])
            ymin = int(ymin * frame.shape[1])
            ymax = int(ymax * frame.shape[1])

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"Class: {class_id}, Score: {score:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
