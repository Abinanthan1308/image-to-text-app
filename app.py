import streamlit as st

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings

warnings.filterwarnings("ignore", message="`resume_download` is deprecated")

model_id = "vikhyatk/moondream2"
revision = "2024-04-02"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

uploaded_file = st.file_uploader("Upload Image")
image = Image.open(uploaded_file)
enc_image = model.encode_image(image)

st.write(model.answer_question(enc_image, "Describe this image.", tokenizer))
