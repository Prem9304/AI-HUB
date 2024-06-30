import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from openai import OpenAI
import requests
import base64
import time
from groq import Groq
from streamlit_option_menu import option_menu

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
nvidia_api_key = os.getenv('NVIDIA_API_KEY')

# App title
st.set_page_config(page_title="AI Project Homepage")

# CSS to make the menu bar smaller
st.markdown("""
    <style>
    .css-18ni7ap.e8zbici2 {
        height: 20px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# Horizontal menu
selected = option_menu(
    None, ["CHAT-AI", "PDF-AI", 'IMG-GEN', 'VID-GEN'], 
    icons=['house', 'cloud-upload', "list-task", 'file-earmark-image', 'camera-reels'], 
    menu_icon="cast", default_index=0, orientation="horizontal"
)

if selected == "CHAT-AI":
    groq_client = Groq(api_key=groq_api_key)

    # Set up NVIDIA client
    nvidia_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=nvidia_api_key
    )

    # List of available NVIDIA models
    nvidia_repo_ids = [
        "mistralai/mixtral-8x7b-instruct-v0.1",
        "meta/llama3-70b-instruct",
        "microsoft/phi-3-mini-4k-instruct"
    ]

    # List of available Hugging Face models
    hf_repo_ids = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "facebook/mbart-large-50-many-to-one-mmt"
    ]

    groq_repo_ids = [
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "whisper-large-v3",
        "gemma-7b-it"
    ]

    # Load the .env file
    dotenv_path = '.env'
    load_dotenv(dotenv_path)

    # Get the Hugging Face API token from the environment
    hf_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

    # Set up the main Streamlit app
    st.title("CHAT-AI")

    # Sidebar configuration
    with st.sidebar:
        st.title('Platform Selection')
        model_selection = st.radio("Choose a Platform", ("Groq", "NVIDIA", "Hugging Face"))

        if model_selection == "Groq":
            st.subheader('Models and parameters')
            repo_id = st.selectbox("Select a Generative Model", groq_repo_ids)
            temperature = st.slider('Temperature', min_value=0.01, max_value=2.0, value=0.1, step=0.01)
            max_length = st.slider('Max Length', min_value=64, max_value=4096, value=512, step=8)

        if model_selection == "NVIDIA":
            st.subheader('Models and parameters')
            repo_id = st.selectbox("Select a Generative Model", nvidia_repo_ids)
            temperature = st.slider('Temperature', min_value=0.01, max_value=2.0, value=0.1, step=0.01)
            top_p = st.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
            max_length = st.slider('Max Length', min_value=64, max_value=4096, value=512, step=8)

        elif model_selection == "Hugging Face":
            st.subheader('Models and parameters')
            repo_id = st.selectbox("Select a Hugging Face Model", hf_repo_ids)
            temperature = st.slider('Temperature', min_value=0.01, max_value=2.0, value=0.7, step=0.01)
            max_length = st.slider('Max Length', min_value=64, max_value=4096, value=512, step=8)

            if hf_api_token is None:
                st.error("API token not found in environment. Please set HUGGINGFACEHUB_API_TOKEN in your .env file.")
                st.stop()

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {"Groq": [], "NVIDIA": [], "Hugging Face": []}

    # Display or clear chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        st.session_state.chat_history = {"Groq": [], "NVIDIA": [], "Hugging Face": []}
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history, key="chat_clear")

    # User-provided prompt
    if prompt := st.chat_input("Enter your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if model_selection == "Groq":
            # Generate response using Groq model
            chat_completion = groq_client.chat.completions.create(
                messages=st.session_state.chat_history["Groq"] + [{"role": "user", "content": prompt}],
                model=repo_id,
                max_tokens=max_length,
                temperature=temperature
            )
            response = chat_completion.choices[0].message.content
            st.session_state.chat_history["Groq"].append({"role": "user", "content": prompt})
            st.session_state.chat_history["Groq"].append({"role": "assistant", "content": response})

        elif model_selection == "NVIDIA":
            # Directly query the NVIDIA model
            completion = nvidia_client.chat.completions.create(
                model=repo_id,
                messages=st.session_state.chat_history["NVIDIA"] + [{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_length
            )

            response_text = completion.choices[0].message.content
            st.session_state.chat_history["NVIDIA"].append({"role": "user", "content": prompt})
            st.session_state.chat_history["NVIDIA"].append({"role": "assistant", "content": response_text})
            response = response_text

        elif model_selection == "Hugging Face":
            # Create the HuggingFaceEndpoint instance
            llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                max_length=max_length,
                temperature=temperature,
                token=hf_api_token
            )

            # Directly query the Hugging Face model
            response = llm.invoke(prompt)
            st.session_state.chat_history["Hugging Face"].append({"role": "user", "content": prompt})
            st.session_state.chat_history["Hugging Face"].append({"role": "assistant", "content": response})

        # Display assistant's response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = response if isinstance(response, str) else ''.join(response)
            placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


elif selected == "PDF-AI":
    st.title("PDF-AI")

    # Sidebar configuration
    with st.sidebar:
        st.title('PDF-AI')
        def vector_embedding(pdf_files):
            if "vectors" not in st.session_state:
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.session_state.docs = []
                for pdf_file in pdf_files:
                    with open(f"./uploaded_pdfs/{pdf_file.name}", "wb") as f:
                        f.write(pdf_file.getbuffer())
                    st.session_state.docs.extend(PyPDFDirectoryLoader("./uploaded_pdfs").load()) ## Document Loading
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) ## Chunk Creation
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) #vector OpenAI embeddings


        st.subheader('Document Upload')
        pdf_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])
        if st.button("Documents Embedding") and pdf_files:
            if not os.path.exists("./uploaded_pdfs"):
                os.makedirs("./uploaded_pdfs")
            vector_embedding(pdf_files)
            st.write("Vector Store DB Is Ready")

    # Define the vector_embedding function
    
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    # User-provided prompt
    if prompt1 := st.chat_input("Enter your question about the documents here..."):
        st.session_state.messages.append({"role": "user", "content": prompt1})
        with st.chat_message("user"):
            st.write(prompt1)

        if "vectors" in st.session_state:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write("Response time:", time.process_time() - start)
            st.write(response['answer'])

            # With a streamlit expander
            with st.expander("Document Similarity Search"):
                # Find the relevant chunks
                for doc in response["context"]:
                    st.write(doc.page_content)
                    st.write("--------------------------------")



elif selected == "IMG-GEN":
    st.title("AI Image Generation with NVIDIA")

    # Sidebar configuration
    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium"
    headers = {
        "Authorization": "Bearer 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",
        "Accept": "application/json",
    }

# Function to make the POST request and display the image
    with st.sidebar:
        st.title('NVIDIA Image Generation')
        st.subheader('Parameters')
        aspect_ratio = st.selectbox("Aspect Ratio", ["16:9", "1:1", "3:2", "4:5", "5:4",])
        negative_prompt= st.text_input("Negative Prompt", "Something You don't want")
        steps= st.slider("Steps", 1, 100, 50)

    def generate_image(prompt):
        payload = {
            "prompt": prompt,
            "cfg_scale": 5,
            "aspect_ratio": aspect_ratio,
            "seed": 0,
            "steps": steps,
            "negative_prompt": negative_prompt
        }

        try:
            # Make the POST request
            response = requests.post(invoke_url, headers=headers, json=payload)
            response.raise_for_status()
            response_body = response.json()

            # Extract the image data (base64 encoded)
            image_data_base64 = response_body.get("image", "")

            # Decode base64 to bytes
            image_data = base64.b64decode(image_data_base64)

            # Display the image in Streamlit
            st.image(image_data, caption='Generated Image', use_column_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")

    prompt = st.text_area('Enter your prompt here:', 'Vintage, steampunk-inspired airship soaring above a sprawling, Victorian-era cityscape, with intricate clockwork mechanisms and steam-powered engines')
    if st.button('Generate Image'):
        generate_image(prompt)

elif selected == "VID-GEN":
    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-video-diffusion"
    headers = {
        "Authorization": "Bearer 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",
        "Accept": "application/json",
    }

    # Streamlit UI
    st.title('Image To Video')
    st.write("Upload an image and click 'Generate Video'")

    # Input field for uploading image and submit button
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    cfg_scale = st.slider('Scale Factor', min_value=1.0, max_value=5.0, value=2.5, step=0.1)
    seed = st.number_input('Seed', value=0, step=1)
    if st.button('Generate Video') and uploaded_file:

        # Read and encode uploaded image as base64
        image_content = uploaded_file.read()
        image_base64 = base64.b64encode(image_content).decode('utf-8')

        # Prepare payload with uploaded image and user inputs
        payload = {
            "image": f"data:image/jpeg;base64,{image_base64}",
            "cfg_scale": cfg_scale,
            "seed": seed
        }

        try:
            # Make POST request to NVIDIA AI API
            response = requests.post(invoke_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise error for bad response status

            # Decode response JSON and extract video data
            response_body = response.json()
            video_base64 = response_body.get("video", "")

            # Convert base64 video data to bytes
            video_bytes = base64.b64decode(video_base64)

            # Display video in Streamlit
            st.video(video_bytes, format='video/mp4')

        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
        