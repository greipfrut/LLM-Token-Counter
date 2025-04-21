import streamlit as st
import anthropic
import json
import os
from huggingface_hub import login
from transformers import AutoTokenizer

st.set_page_config(page_title="LLM Token Counter", page_icon="🤖", layout="wide")

st.title("🎈 LLM Token Counter")
st.markdown(
    "This app counts tokens for different language models based on your input text."
)

# Tabs for model provider selection
provider_tab = st.tabs(["Anthropic Models", "Hugging Face Models"])

with provider_tab[0]:  # Anthropic Models
    st.header("Anthropic (Claude) Models")

    # API key input (with warning about security)
    anthropic_key = st.text_input(
        "Enter your Anthropic API Key",
        type="password",
        help="⚠️ Never share your API key. Leave empty to use ANTHROPIC_API_KEY environment variable.",
    )

    # If no key provided, try to get from environment
    if not anthropic_key:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # Model selection for Anthropic
    anthropic_model_options = {
        "Claude 3.7 Sonnet": "claude-3-7-sonnet-20250219",
        "Claude 3.5 Sonnet": "claude-3-5-sonnet-20240620",
        "Claude 3.5 Haiku": "claude-3-5-haiku-20240307",
        "Claude 3 Haiku": "claude-3-haiku-20240307",
        "Claude 3 Opus": "claude-3-opus-20240229",
    }

    selected_anthropic_model = st.selectbox(
        "Select Claude Model", list(anthropic_model_options.keys())
    )

    # System message (optional)
    st.subheader("System Message (Optional)")
    system_message = st.text_area(
        "System Message", placeholder="e.g., You are a helpful assistant", height=100
    )

    # User message input
    st.subheader("Message Content")
    anthropic_user_message = st.text_area(
        "Enter your message here",
        placeholder="Hello, Claude! How are you today?",
        height=200,
        key="anthropic_message",
    )

    # Button to count tokens for Anthropic
    if st.button("Count Tokens (Anthropic)"):
        if not anthropic_key:
            st.error(
                "No Anthropic API key found. Please enter a key or set the ANTHROPIC_API_KEY environment variable."
            )
        elif not anthropic_user_message:
            st.warning("Please enter a message to count tokens")
        else:
            try:
                # Initialize client with API key
                client = anthropic.Anthropic(api_key=anthropic_key)

                # Create the request
                count_request = {
                    "model": anthropic_model_options[selected_anthropic_model],
                    "messages": [{"role": "user", "content": anthropic_user_message}],
                }

                # Add system message if provided
                if system_message:
                    count_request["system"] = system_message

                # Make the API call to count tokens
                response = client.messages.count_tokens(**count_request)

                # Display results
                st.success(f"Input tokens: {response.input_tokens}")

                # Display the full JSON response in an expandable section
                with st.expander("View Full API Response"):
                    st.code(
                        json.dumps(response.model_dump(), indent=2), language="json"
                    )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with provider_tab[1]:  # Hugging Face Models
    st.header("Hugging Face Models")

    # HF Token input
    hf_token = st.text_input(
        "Enter your Hugging Face Token",
        type="password",
        help="⚠️ Never share your token. Leave empty to use HF_TOKEN environment variable.",
    )

    # If no token provided, try to get from environment
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN", "")

    # Login status tracker
    if "hf_logged_in" not in st.session_state:
        st.session_state.hf_logged_in = False

    # Login button
    if not st.session_state.hf_logged_in and st.button("Login to Hugging Face"):
        if not hf_token:
            st.error(
                "No Hugging Face token found. Please enter a token or set the HF_TOKEN environment variable."
            )
        else:
            try:
                login(token=hf_token)
                st.session_state.hf_logged_in = True
                st.success("Successfully logged in to Hugging Face")
            except Exception as e:
                st.error(f"Login failed: {str(e)}")

    if st.session_state.hf_logged_in or hf_token:
        # Predefined popular models
        hf_model_options = [
            "mistralai/Mistral-Small-24B-Instruct-2501",
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            "google/codegemma-7b",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "microsoft/Phi-4-multimodal-instruct",
            "nvidia/Llama-3.3-70B-Instruct-FP4",
            "Other (specify)",
        ]

        selected_hf_model = st.selectbox("Select Hugging Face Model", hf_model_options)

        # Custom model input
        if selected_hf_model == "Other (specify)":
            custom_hf_model = st.text_input(
                "Enter model name (e.g., organization/model-name)"
            )
            selected_hf_model = (
                custom_hf_model if custom_hf_model else "gpt2"
            )  # Default to gpt2 if empty

        # User message input for HF
        hf_user_message = st.text_area(
            "Enter your message here",
            placeholder="Hello, world!",
            height=200,
            key="hf_message",
        )

        # Button to count tokens for HF
        if st.button("Count Tokens (Hugging Face)"):
            if not hf_user_message:
                st.warning("Please enter a message to count tokens")
            else:
                try:
                    with st.spinner(f"Loading tokenizer for {selected_hf_model}..."):
                        tokenizer = AutoTokenizer.from_pretrained(selected_hf_model)

                    # Count tokens in different ways
                    tokens = tokenizer.tokenize(hf_user_message)
                    token_ids = tokenizer.encode(hf_user_message)

                    # Display results
                    st.success(f"Token count: {len(tokens)}")
                    st.success(f"Token IDs count: {len(token_ids)}")

                    # Show the actual tokens
                    with st.expander("View Token Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Tokens")
                            st.json([f"{i}: {token}" for i, token in enumerate(tokens)])
                        with col2:
                            st.subheader("Token IDs")
                            st.json(
                                [
                                    f"{i}: {token_id}"
                                    for i, token_id in enumerate(token_ids)
                                ]
                            )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Additional information
with st.expander("About Token Counting"):
    st.markdown("""
    ### What are tokens?

    Tokens are chunks of text that language models process. They can be parts of words, whole words, 
    or even punctuation. Different models tokenize text differently.

    ### Why count tokens?

    - **Cost Management**: Understanding token usage helps manage API costs
    - **Model Limitations**: Different models have different token limits
    - **Performance Optimization**: Helps optimize prompts for better responses

    ### Token Counting Tips

    - Shorter messages use fewer tokens
    - Special formatting, code blocks, and unusual characters may use more tokens
    - For Claude models, the system message also counts toward your token usage
    - Hugging Face models may tokenize text differently than Anthropic models
    """)

# Footer
st.markdown("---")
st.markdown("Created with Streamlit, Anthropic API, and Hugging Face Transformers")
