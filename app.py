import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Constants
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

BLACK = (0,0,0)
BLUE = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)

# Load classes
@st.cache_resource
def load_classes():
    classesFile = "coco.names"
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes

# Load model
@st.cache_resource
def load_model():
    model_path = "models/yolov5s.onnx"
    session = ort.InferenceSession(model_path)
    return session

# Draw label helper
def draw_label(input_image, label, left, top):
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

# Preprocess image
def pre_process(input_image):
    image = cv2.resize(input_image, (INPUT_WIDTH, INPUT_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)   # add batch dimension
    return image

# Post-process outputs
def post_process(input_image, outputs, classes):
    outputs = outputs[0]
    class_ids = []
    confidences = []
    boxes = []

    rows = outputs.shape[1]
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = outputs[0][r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices.flatten():
        box = boxes[i]
        left, top, width, height = box
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        draw_label(input_image, label, left, top)

    return input_image

# Video processor for real-time webcam
class YOLOObjectDetector(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.classes = load_classes()
        self.input_name = self.model.get_inputs()[0].name
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame
        input_tensor = pre_process(img)
        outputs = self.model.run(None, {self.input_name: input_tensor})
        processed_img = post_process(img.copy(), outputs, self.classes)
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

def main():
    st.title("YOLOv5 Object Detection with ONNX Runtime")
    
    # Load resources
    classes = load_classes()
    session = load_model()
    input_name = session.get_inputs()[0].name
    
    # Create tabs for different functions
    tab1, tab2 = st.tabs(["Image Upload", "Webcam Real-time"])
    
    with tab1:
        st.header("Object Detection from Uploaded Images")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            frame = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            st.image(frame, channels="BGR", caption="Uploaded Image", use_container_width=True)

            input_tensor = pre_process(frame)

            # Run inference
            outputs = session.run(None, {input_name: input_tensor})

            # Draw detections
            img = post_process(frame.copy(), outputs, classes)

            # Display results
            st.image(img, channels="BGR", caption="Detected Objects", use_container_width=True)
    
    with tab2:
        st.header("Real-time Object Detection")
        st.write("Click on 'Start' to enable your webcam and begin real-time object detection")
        
        # WebRTC configuration
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Start the WebRTC streamer
        webrtc_streamer(
            key="yolov5-object-detection",
            video_processor_factory=YOLOObjectDetector,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        st.markdown("""
        ### Notes:
        - Make sure to allow browser access to your webcam
        - For best performance, ensure good lighting
        - Detection may slow down on lower-end devices
        """)

if __name__ == "__main__":
    main()