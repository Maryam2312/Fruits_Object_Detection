import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

model = YOLO('best.pt')

st.title('Fruits Object Detection with YOLO')

file = st.file_uploader("Choose an Image", type=['jpg', 'png'])

if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    st.write("")

    np_image = np.array(image)
    opencv_image = np_image[:, :, ::-1].copy()  # Convert RGB to BGR

    if st.button('Analyse Image'):

        predictions = model(opencv_image)

        # Extract detected items
        items = []
        for pred in predictions:
            for box in pred.boxes:
                clss = int(box.cls)
                label = pred.names[clss]
                items.append(label)

                # Draw bounding boxes and labels on the image
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(opencv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(opencv_image, f'{label} {confidence:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Convert image back to RGB for displaying with Streamlit
        image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption='Image with Detections', use_column_width=True)

        # Display detected items
        st.write("Detected items:")
        st.write(items)
