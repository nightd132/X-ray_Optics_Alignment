import streamlit as st
import torch
from pathlib import Path
from fabio.fabioutils import BytesIO
from src.optics_alignment_regression.predict import OpticsAlignmentPredictor
from src.optics_alignment_regression.utils import find_best_experiment
from configs.configs import EXPERIMENT_SUMMARY_CSV_PATH, CHECKPOINT_DIR

st.title("X-Ray Optics Alignment")

@st.cache_resource
def load_predictor():
    best_experiment = find_best_experiment(EXPERIMENT_SUMMARY_CSV_PATH)
    ckt_path = CHECKPOINT_DIR/ best_experiment["experiment"] / "best.pt"
    st.text(ckt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return OpticsAlignmentPredictor(ckt_path, device)

uploaded_file = st.file_uploader("Choose the file image", type=["cbf"])
predictor = load_predictor()
if uploaded_file is not None:
    if st.button("Predict"):
        stream = BytesIO(uploaded_file.getvalue())
        predict_result = predictor.predict(stream)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Image")
            st.image(predict_result["raw_image"])
        with col2:
            st.subheader("Preprocessed Image")
            st.image(predict_result["preprocessed_image"])

        st.subheader("Predicted Offset x:")
        st.metric(label="Offset x", value=f"{predict_result['predicted_offset_x']}")
        st.metric(label="Offset z", value=f"{predict_result['predicted_offset_z']}")

