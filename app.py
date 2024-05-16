import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import torch
import base64
import sentencepiece
import accelerate

# Verify the versions of sentencepiece and accelerate
print(sentencepiece.__version__)
print(accelerate.__version__)

# Model and tokenizer
checkpoint = 't5-small'
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint,
    device_map='auto',
    torch_dtype=torch.float32
)
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
print("Model and tokenizer loaded successfully")

# File loader
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts += text.page_content
    return final_texts

# LM pipeline
def llm_pipeline(filepath, max_length=500, min_length=100):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=max_length,
        min_length=min_length
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    return result[0]['summary_text']

@st.cache_resource
def displayPDF(File):
    with open(File, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code
st.set_page_config(layout='wide')

def main():
    st.title('Document Summarizer')
    uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf', 'doc'])

    # Input fields for customization
    max_length = st.sidebar.slider("Max Length", min_value=100, max_value=1000, value=500, step=50)
    min_length = st.sidebar.slider("Min Length", min_value=50, max_value=500, value=100, step=50)

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        filepath = "data/" + uploaded_file.name
        with open(filepath, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())
        with col1:
            st.info("Uploaded PDF File")
            displayPDF(filepath)
        with col2:
            st.info("Summarization below")
            if st.button("Summarize"):
                summary = llm_pipeline(filepath, max_length=max_length, min_length=min_length)
                st.success(summary)


if __name__ == '__main__':
    main()
