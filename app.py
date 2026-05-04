from statistics import mean
from typing import get_args

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from fast_alpr import ALPR
from fast_alpr.default_detector import PlateDetectorModel
from fast_alpr.default_ocr import OcrModel

# Default models
DETECTOR_MODELS = list(get_args(PlateDetectorModel))
OCR_MODELS = list(get_args(OcrModel))

# Put global OCR first
OCR_MODELS.remove("cct-s-v2-global-model")
OCR_MODELS.insert(0, "cct-s-v2-global-model")


@st.cache_resource
def get_alpr(detector_model: PlateDetectorModel, ocr_model: OcrModel) -> ALPR:
    return ALPR(detector_model=detector_model, ocr_model=ocr_model)

st.title("FastALPR Demo")
st.write("An automatic license plate recognition (ALPR) system with customizable detector and OCR models.")
st.markdown(
    """
[FastALPR](https://github.com/ankandrew/fast-alpr) uses
[open-image-models](https://github.com/ankandrew/open-image-models)
for plate detection and
[fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr)
for optical character recognition (**OCR**).
"""
)

detector_model = st.sidebar.selectbox("Choose Detector Model", DETECTOR_MODELS)
ocr_model = st.sidebar.selectbox("Choose OCR Model", OCR_MODELS)

uploaded_file = st.file_uploader(
    "Upload an image of a vehicle with a license plate",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    rgb_array = np.array(img)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    alpr = get_alpr(detector_model=detector_model, ocr_model=ocr_model)

    with st.spinner("Processing..."):
        drawn = alpr.draw_predictions(bgr_array.copy())

    annotated_rgb = cv2.cvtColor(drawn.image, cv2.COLOR_BGR2RGB)
    annotated_img = Image.fromarray(annotated_rgb)
    st.image(annotated_img, caption="Annotated Image with OCR Results", use_container_width=True)

    if drawn.results:
        st.write("**OCR Results:**")
        for result in drawn.results:
            if result.ocr is None:
                st.write("- Detected plate with no OCR result")
                continue

            confidence = (
                mean(result.ocr.confidence)
                if isinstance(result.ocr.confidence, list)
                else result.ocr.confidence
            )
            text = result.ocr.text or "N/A"
            details = f"- Detected Plate: `{text}` with confidence `{confidence:.2f}`"
            if result.ocr.region:
                details += f" in region `{result.ocr.region}`"
                if result.ocr.region_confidence is not None:
                    details += f" (`{result.ocr.region_confidence:.2f}`)"
            st.write(details)
    else:
        st.write("No license plate detected.")
else:
    st.write("Please upload an image to continue.")