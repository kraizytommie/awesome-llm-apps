import streamlit as st
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
import tempfile
import os
import requests
from datetime import datetime
import docx
import pandas as pd
from PIL import Image
import json
import time

class DocumentParser:
    @staticmethod
    def parse_pdf(file_path):
        # PDF parsing logic here
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')

    @staticmethod
    def parse_docx(file_path):
        doc = docx.Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

    @staticmethod
    def parse_excel(file_path):
        df = pd.read_excel(file_path)
        return df.to_string()

    @staticmethod
    def parse_image(file_path):
        image = Image.open(file_path)
        # Add OCR logic here if needed
        return f"Image processed: {file_path}"

class DeepSeekChat:
    def __init__(self, model="anthropic/claude-2", api_key=None):
        self.model = model
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/kraizytommie/litigatingllm",
            "X-Title": "Legal Document Analyzer"
        }
        if st.session_state.debug_mode:
            st.write("Debug: Initialized DeepSeekChat with model:", model)

    def query(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                st.write(f"Debug: Attempt {attempt + 1} of {max_retries}")
                
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
                
                if st.session_state.debug_mode:
                    st.write("Debug: Sending request with payload:", json.dumps(payload, indent=2))
                    st.write("Debug: Using headers:", {k: v if k != "Authorization" else "[HIDDEN]" for k, v in self.headers.items()})
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if st.session_state.debug_mode:
                    st.write(f"Debug: Response status code: {response.status_code}")
                    if response.status_code != 200:
                        st.write("Debug: Error response:", response.text)

                if response.status_code == 401:
                    return "Error: Invalid API key. Please check your OpenRouter API key."
                elif response.status_code == 400:
                    error_msg = response.json().get('error', {}).get('message', 'Bad request')
                    return f"Error: {error_msg}"
                
                response.raise_for_status()
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return "Error: No response content received from API"
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    st.error(f"API Error: {str(e)}")
                    return f"Failed to get response from OpenRouter API: {str(e)}"
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Unexpected Error: {str(e)}")
                    return f"An unexpected error occurred: {str(e)}"
                time.sleep(2 ** attempt)  # Exponential backoff

def init_session_state():
    """Initialize session state variables"""
    if 'openrouter_api_key' not in st.session_state:
        st.session_state.openrouter_api_key = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "anthropic/claude-2"
    if 'legal_team' not in st.session_state:
        st.session_state.legal_team = None
    if 'deepseek_chat' not in st.session_state:
        st.session_state.deepseek_chat = None
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

def process_documents(uploaded_files):
    """Process multiple documents and initialize DeepSeek chat"""
    if not st.session_state.openrouter_api_key:
        raise ValueError("OpenRouter API key not provided")
    
    processed_docs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            try:
                # Save file to temporary directory
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Extract text content based on file type
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                content = ""

                if file_ext == '.pdf':
                    if st.session_state.debug_mode:
                        st.write(f"Debug: Processing PDF file: {uploaded_file.name}")
                    content = DocumentParser.parse_pdf(file_path)
                elif file_ext == '.docx':
                    if st.session_state.debug_mode:
                        st.write(f"Debug: Processing DOCX file: {uploaded_file.name}")
                    content = DocumentParser.parse_docx(file_path)
                elif file_ext in ['.xlsx', '.xls']:
                    if st.session_state.debug_mode:
                        st.write(f"Debug: Processing Excel file: {uploaded_file.name}")
                    content = DocumentParser.parse_excel(file_path)
                elif file_ext in ['.jpg', '.jpeg', '.png']:
                    if st.session_state.debug_mode:
                        st.write(f"Debug: Processing Image file: {uploaded_file.name}")
                    content = DocumentParser.parse_image(file_path)
                else:
                    if st.session_state.debug_mode:
                        st.write(f"Debug: Processing text file: {uploaded_file.name}")
                    with open(file_path, 'r') as f:
                        content = f.read()

                if content:
                    doc_info = {
                        'name': uploaded_file.name,
                        'content': content,
                        'type': file_ext,
                        'timestamp': datetime.now().isoformat(),
                        'path': file_path
                    }
                    processed_docs.append(doc_info)
                    if st.session_state.debug_mode:
                        st.write(f"Debug: Successfully processed {uploaded_file.name}")
                else:
                    st.warning(f"No content extracted from {uploaded_file.name}")

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                if st.session_state.debug_mode:
                    st.exception(e)
                continue

    if processed_docs:
        st.session_state.documents.extend(processed_docs)
        # Initialize DeepSeek chat with selected model
        if not st.session_state.deepseek_chat:
            if st.session_state.debug_mode:
                st.write(f"Debug: Initializing chat with model {st.session_state.selected_model}")
            st.session_state.deepseek_chat = DeepSeekChat(
                model=st.session_state.selected_model,
                api_key=st.session_state.openrouter_api_key
            )
        return True
    return False

def export_results(analysis_results, format='json'):
    """Export analysis results in various formats"""
    if format == 'json':
        return json.dumps(analysis_results, indent=2)
    elif format == 'csv':
        df = pd.DataFrame(analysis_results)
        return df.to_csv(index=False)
    elif format == 'pdf':
        # Add PDF export logic here
        return "PDF export not implemented yet"
    return None

def create_agent_with_context(name, role, instructions, tools=None, documents=None):
    """Create an agent with document context"""
    if documents:
        doc_context = "\n\n".join([
            f"Document: {doc['name']}\nContent: {doc['content'][:1000]}..." 
            for doc in documents
        ])
        context_instructions = [
            "You have access to the following documents:",
            doc_context,
            "Always reference specific parts of the documents in your analysis.",
            "Provide clear citations when referring to document content."
        ]
        instructions = context_instructions + instructions

    agent_config = {
        "name": name,
        "role": role,
        "instructions": instructions,
        "markdown": True
    }
    if tools:
        agent_config["tools"] = tools

    return Agent(**agent_config)

def initialize_agents():
    """Initialize all specialized agents"""
    if st.session_state.debug_mode:
        st.write("Debug: Initializing agents")

    documents = st.session_state.documents

    agents = {
        'document_parser': create_agent_with_context(
            "Document Parser",
            "Document processing specialist",
            [
                "Extract and structure information from documents",
                "Identify key sections and metadata",
                "Create summaries of document content"
            ],
            documents=documents
        ),
        'legal_researcher': create_agent_with_context(
            "Legal Researcher",
            "Legal research specialist",
            [
                "Find relevant cases and precedents",
                "Analyze legal arguments and citations",
                "Connect document content with legal research"
            ],
            tools=[DuckDuckGo()],
            documents=documents
        ),
        'fraud_detector': create_agent_with_context(
            "Fraud Detector",
            "Fraud detection specialist",
            [
                "Identify potential fraudulent patterns",
                "Flag suspicious activities or claims",
                "Cross-reference document details"
            ],
            documents=documents
        ),
        'strategy_advisor': create_agent_with_context(
            "Strategy Advisor",
            "Legal strategy specialist",
            [
                "Develop comprehensive legal strategies",
                "Suggest counterclaims and motions",
                "Base recommendations on document evidence"
            ],
            documents=documents
        )
    }

    # Create team lead agent
    agents['legal_team'] = create_agent_with_context(
        "Legal Team Lead",
        "Team coordinator",
        [
            "Coordinate analysis between specialized agents",
            "Ensure all recommendations are based on document evidence",
            "Provide comprehensive insights and recommendations"
        ],
        documents=documents
    )

    if st.session_state.debug_mode:
        st.write("Debug: Successfully initialized all agents")

    return agents

def main():
    st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
    init_session_state()

    # Main content
    st.title("AI Legal Agent Team 👨‍⚖️")

    # API Configuration
    with st.sidebar:
        st.header("🔑 API Configuration")
        with st.expander("ℹ️ API Key Help", expanded=False):
            st.markdown("""
            1. Get your OpenRouter API key from [openrouter.ai](https://openrouter.ai)
            2. Enter the key in the field below
            3. Select your preferred model
            4. Your settings are stored securely in the session
            """)
   
        with st.form("api_config"):
            openrouter_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                value=st.session_state.openrouter_api_key if st.session_state.openrouter_api_key else "",
                help="Enter your OpenRouter API key"
            )
            
            st.write("🤖 Model Configuration")
            model_col1, model_col2 = st.columns(2)
            
            with model_col1:
                model_source = st.radio(
                    "Model Source",
                    ["OpenRouter Models", "Custom Endpoint"],
                    help="Choose between pre-configured models or enter a custom endpoint"
                )
            
            with model_col2:
                if model_source == "OpenRouter Models":
                    model_options = {
                        "DeepSeek 67B": "deepseek-ai/deepseek-67b-chat",
                        "DeepSeek Coder": "deepseek-ai/deepseek-coder-33b-instruct",
                        "Mixtral 8x7B": "mistralai/mixtral-8x7b-instruct",
                        "Claude 3 Opus": "anthropic/claude-3-opus",
                        "GPT-4 Turbo": "openai/gpt-4-turbo"
                    }
                    
                    selected_model = st.selectbox(
                        "Select Model",
                        options=list(model_options.keys()),
                        index=list(model_options.values()).index(st.session_state.selected_model) if st.session_state.selected_model in model_options.values() else 0,
                        help="Choose from available models"
                    )
                    final_model = model_options[selected_model]
                else:
                    custom_endpoint = st.text_input(
                        "Custom Model Endpoint",
                        value=st.session_state.selected_model if not any(st.session_state.selected_model in v for v in model_options.values()) else "",
                        help="Enter the full model endpoint (e.g., huggingface/model-name)"
                    )
                    final_model = custom_endpoint
            
            st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
            
            if st.form_submit_button("Save Configuration"):
                st.session_state.openrouter_api_key = openrouter_key
                st.session_state.selected_model = final_model
                st.success(f"✅ Configuration saved! Using model: {final_model}")
                if st.session_state.debug_mode:
                    st.info("Debug mode enabled - you'll see detailed API interaction logs")

        st.divider()

        # Analysis Mode Selection
        st.header("🔍 Analysis Mode")
        analysis_mode = st.selectbox(
            "Choose Analysis Mode",
            ["Document Analysis", "Fraud Detection", "Cross-Reference", "Custom Query"]
        )

    if not st.session_state.openrouter_api_key:
        st.info("👈 Please configure your OpenRouter API key in the sidebar to begin")
        return

    # File Upload Section
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Drag and drop your legal documents",
        type=['pdf', 'docx', 'txt', 'xlsx', 'csv', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Supported formats: PDF, Word, Text, Excel, CSV, and Images"
    )

    if uploaded_files:
        with st.spinner("Processing documents..."):
            try:
                if process_documents(uploaded_files):
                    st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)")
                    
                    # Initialize all agents
                    agents = initialize_agents()
                    st.session_state.legal_team = agents['legal_team']
                    
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

    # Analysis Section
    if st.session_state.documents and st.session_state.legal_team:
        st.header("📊 Analysis Dashboard")

        # Document Overview
        with st.expander("📑 Document Overview", expanded=True):
            for doc in st.session_state.documents:
                st.write(f"📄 {doc['name']} ({doc['type']}) - Uploaded: {doc['timestamp']}")

        # Analysis Options based on mode
        if analysis_mode == "Document Analysis":
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Contract Review", "Legal Research", "Risk Assessment", 
                 "Compliance Check", "Fraud Detection"]
            )
        elif analysis_mode == "Cross-Reference":
            if len(st.session_state.documents) < 2:
                st.warning("Please upload at least 2 documents for cross-referencing")
                return
            docs_to_compare = st.multiselect(
                "Select documents to compare",
                [doc['name'] for doc in st.session_state.documents],
                max_selections=2
            )
        elif analysis_mode == "Custom Query":
            custom_query = st.text_area(
                "Enter your analysis query",
                help="Specify what you want to analyze or investigate"
            )

        # Analysis Execution
        if st.button("Run Analysis"):
            with st.spinner("Analyzing documents..."):
                try:
                    # Prepare analysis query based on mode
                    if analysis_mode == "Document Analysis":
                        query = f"Analyze the documents focusing on {analysis_type}"
                    elif analysis_mode == "Cross-Reference":
                        query = f"Compare and find inconsistencies between the selected documents"
                    else:
                        query = custom_query

                    # Run analysis using DeepSeek
                    response = st.session_state.deepseek_chat.query(query)
                    
                    # Display results in tabs
                    tabs = st.tabs(["Analysis", "Key Points", "Recommendations", "Export"])
                    
                    with tabs[0]:
                        st.markdown("### Detailed Analysis")
                        st.markdown(response)
                    
                    with tabs[1]:
                        st.markdown("### Key Points")
                        key_points = st.session_state.deepseek_chat.query(
                            f"Summarize the key points from this analysis: {response}"
                        )
                        st.markdown(key_points)
                    
                    with tabs[2]:
                        st.markdown("### Recommendations")
                        recommendations = st.session_state.deepseek_chat.query(
                            f"Provide specific recommendations based on this analysis: {response}"
                        )
                        st.markdown(recommendations)
                    
                    with tabs[3]:
                        st.markdown("### Export Results")
                        export_format = st.selectbox(
                            "Select export format",
                            ["JSON", "CSV", "PDF"]
                        )
                        if st.button("Export"):
                            results = {
                                "analysis": response,
                                "key_points": key_points,
                                "recommendations": recommendations,
                                "timestamp": datetime.now().isoformat()
                            }
                            exported = export_results(results, export_format.lower())
                            if exported:
                                st.download_button(
                                    "Download Results",
                                    exported,
                                    f"analysis_results.{export_format.lower()}",
                                    mime=f"application/{export_format.lower()}"
                                )

                    # Save to analysis history
                    st.session_state.analysis_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "mode": analysis_mode,
                        "query": query,
                        "results": response
                    })

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

        # Analysis History
        if st.session_state.analysis_history:
            st.header("📜 Analysis History")
            for analysis in reversed(st.session_state.analysis_history):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(f"🕒 {analysis['timestamp']}")
                    st.write(f"Mode: {analysis['mode']}")
                with col2:
                    st.markdown("**Results:**")
                    st.markdown(analysis['results'])
                st.divider()

if __name__ == "__main__":
    main()