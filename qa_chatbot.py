from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams 
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames 
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
import os

import gradio as gr 

def warn(*args,**kwargs):
    pass 
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

## LLM
def get_llm():
    model_id = 'ibm/granite-3-2-8b-instruct'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256, 
        GenParams.TEMPERATURE: 0.5, 
    }

    project_id = 'skills-network'

    api_key = os.getenv("WATSONX_API_KEY") or os.getenv("WATSONX_TOKEN")
    if not api_key:
        raise RuntimeError(
            "Missing Watsonx credentials. Set environment variable WATSONX_API_KEY or WATSONX_TOKEN."
        )

    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url='https://us-south.ml.cloud.ibm.com',
        project_id=project_id,
        params=parameters,
        api_key=api_key,
        )
    return watsonx_llm

def document_loader(file):
    # Accept either a filepath string (Gradio with type="filepath") or a file-like object
    if isinstance(file, str):
        path = file
    else:
        path = getattr(file, "name", file)
    loader = PyPDFLoader(path)
    loaded_document = loader.load()
    return loaded_document

#Text splitter
def text_splitter(data):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
        )
        chunks = splitter.split_documents(data)
        return chunks

## Embedding modele
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding

## Vector db
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding=embedding_model)
    return vectordb

## Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

## QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    return_source_documents=True)
    # Different LangChain versions expose different call APIs; try the common ones
    try:
        return qa.run(query)
    except Exception:
        try:
            response = qa({"query": query})
            if isinstance(response, dict) and "result" in response:
                return response["result"]
            return str(response)
        except Exception as e:
            return f"QA call failed: {e}"

# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title='AI engineering Chat bot',
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

# Launch the app
rag_application.launch(server_name='127.0.0.1', server_port= 7860)