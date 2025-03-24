import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from huggingface_hub import login

# Set page title and configuration
st.set_page_config(
    page_title="Language Model Evaluation",
    page_icon="🤖",
    layout="centered"
)

# Language selection
lang_option = st.sidebar.radio(
    "Language / اللغة",
    ("العربية", "English"),
    index=0
)

is_arabic = lang_option == "العربية"

# Title with proper language-based direction
title_style = "text-align: center; direction: rtl;" if is_arabic else "text-align: center; direction: ltr;"
if is_arabic:
    st.markdown(f"<h1 style='{title_style}'>تقييم فهم النموذج اللغوي</h1>", unsafe_allow_html=True)
else:
    st.markdown(f"<h1 style='{title_style}'>Language Model Evaluation</h1>", unsafe_allow_html=True)

# API Key Configuration
with st.sidebar:
    st.markdown("---")
    
    if is_arabic:
        st.markdown("## مفتاح API:")
        api_key_help = "مطلوب لاستخدام النماذج الكبيرة مثل Llama-2 و Mistral-7B"
    else:
        st.markdown("## API Key:")
        api_key_help = "Required for using large models like Llama-2 and Mistral-7B"
    
    # API key input with password masking
    api_key = st.text_input("Hugging Face API Token", type="password", help=api_key_help)
    
    # Apply API key if provided
    if api_key:
        try:
            # Check if API key is valid by attempting to use it
            login(token=api_key)
            # If we reach here, login succeeded
            st.success("✅ API Key configured" if not is_arabic else "✅ تم تكوين مفتاح API")
            has_api_key = True
            # Set environment variable for the token
            os.environ["HUGGINGFACE_TOKEN"] = api_key
        except Exception as e:
            st.error(f"API Key Error: {str(e)}" if not is_arabic else f"خطأ في مفتاح API: {str(e)}")
            has_api_key = False
    else:
        has_api_key = False
        st.warning("No API key provided. Some models may not be available." if not is_arabic else "لم يتم توفير مفتاح API. قد لا تتوفر بعض النماذج.")

    st.markdown("---")

# Sidebar for model selection
with st.sidebar:
    # Model selection header
    if is_arabic:
        st.markdown("## اختر النموذج اللغوي:")
    else:
        st.markdown("## Select Language Model:")
    
    # Define model options - always include all models
    small_models = [
        "facebook/opt-125m",
        "distilgpt2",
        "EleutherAI/gpt-neo-125m",
        "microsoft/DialoGPT-small"
    ]
    
    large_models = [
        "facebook/opt-1.3b",
        "bigscience/bloom-560m"
    ]
    
    # Always include all models, but mark large models specially
    model_options = small_models + large_models
    
    # Define all model labels with special indicators for large models
    model_labels = {
        "facebook/opt-125m": "Facebook OPT-125M (Small)",
        "distilgpt2": "DistilGPT-2 (Small)",
        "EleutherAI/gpt-neo-125m": "GPT-Neo-125M (Small)",
        "microsoft/DialoGPT-small": "DialoGPT (Small)",
        "facebook/opt-1.3b": "Facebook OPT-1.3B (Medium) 🔑" if not has_api_key else "Facebook OPT-1.3B (Medium)",
        "bigscience/bloom-560m": "BLOOM-560M (Medium) 🔑" if not has_api_key else "BLOOM-560M (Medium)"
    }
    
    # Default to distilgpt2 (index 1) which is the most reliable
    default_index = 1
    
    model_option = st.selectbox(
        "",
        model_options,
        format_func=lambda x: model_labels[x],
        index=default_index
    )
    
    # Info about models
    if is_arabic:
        if "llama" in model_option.lower() or "mistral" in model_option.lower() or "opt-1.3b" in model_option.lower() or "bloom" in model_option.lower():
            if has_api_key:
                st.info("هذا نموذج كبير وقد يستغرق وقتًا أطول للتحميل والتنفيذ.")
            else:
                st.warning("يتطلب هذا النموذج مفتاح API صالح من Hugging Face. يرجى إدخال المفتاح أعلاه.")
        else:
            st.info("تم استخدام نماذج صغيرة وفعالة للتوافق مع قيود الموارد.")
    else:
        if "llama" in model_option.lower() or "mistral" in model_option.lower() or "opt-1.3b" in model_option.lower() or "bloom" in model_option.lower():
            if has_api_key:
                st.info("This is a large model and may take longer to load and run.")
            else:
                st.warning("This model requires a valid Hugging Face API key. Please enter your key above.")
        else:
            st.info("Small, efficient models are used due to resource constraints.")

# Function to set environment variable for API key
def setup_env_for_api_key():
    if api_key:
        os.environ["HUGGINGFACE_TOKEN"] = api_key
        return True
    return False

# Cache the model loading to improve performance
@st.cache_resource
def load_model(model_name, token=None):
    try:
        # Set API token for authentication
        auth_token = token if token else os.environ.get("HUGGINGFACE_TOKEN", None)
        
        # For larger models, we need to adjust the loading parameters
        is_large_model = "llama" in model_name.lower() or "mistral" in model_name.lower() or "opt-1.3b" in model_name.lower() or "bloom" in model_name.lower()
        
        # Tokenizer parameters
        tokenizer_params = {
            "trust_remote_code": True,
        }
        
        # For large models, ensure we always pass the token
        if is_large_model and auth_token:
            tokenizer_params["token"] = auth_token
        elif is_large_model and not auth_token:
            # If large model but no token, raise error
            raise ValueError(f"API token required to load {model_name}")
            
        # Load tokenizer with appropriate parameters
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                **tokenizer_params
            )
        except Exception as tokenizer_error:
            st.error(f"Failed to load tokenizer: {str(tokenizer_error)}")
            # Fallback to distilgpt2 if the selected tokenizer fails
            tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            
        # Model parameters
        load_params = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # For large models, ensure we always pass the token
        if is_large_model and auth_token:
            load_params["token"] = auth_token
        
        # Add model-specific parameters
        if is_large_model:
            load_params["torch_dtype"] = torch.float32  # Use float32 for better compatibility
            # Only use device_map if CUDA is available
            if torch.cuda.is_available():
                load_params["device_map"] = "auto"
                load_params["max_memory"] = {0: "12GB"}
        else:
            if torch.cuda.is_available():
                load_params["torch_dtype"] = torch.float16
                load_params["device_map"] = "auto"
        
        # Load model with appropriate parameters
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_params)
            return tokenizer, model
        except Exception as model_error:
            st.error(f"Failed to load model: {str(model_error)}")
            # Fallback to distilgpt2 if the selected model fails
            model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            return tokenizer, model
            
    except Exception as e:
        st.error(f"Error in model loading process: {str(e)}")
        # Fallback to distilgpt2 as a last resort
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        return tokenizer, model

# Set text direction based on language
text_direction = "rtl" if is_arabic else "ltr"

# Main content with language-specific labels and proper text direction
if is_arabic:
    st.markdown(f"<div style='direction: {text_direction};'><h3>أدخل سؤالاً أو تعليمات:</h3></div>", unsafe_allow_html=True)
    placeholder_text = "اكتب هنا..."
    loading_text = "جاري تحميل النموذج... قد يستغرق هذا بضع دقائق."
    button_text = "توليد الرد"
    warning_text = "الرجاء إدخال نص"
    generating_text = "جاري توليد الرد..."
    response_header = "<div style='direction: rtl;'><h3>الرد:</h3></div>"
    evaluation_header = "<div style='direction: rtl;'><h3>التقييم:</h3></div>"
    understood_text = "✅ فهمه"
    not_understood_text = "❌ لم يفهمه"
else:
    st.markdown(f"<div style='direction: {text_direction};'><h3>Enter a question or instructions:</h3></div>", unsafe_allow_html=True)
    placeholder_text = "Type here..."
    loading_text = "Loading model... This may take a few minutes."
    button_text = "Generate Response"
    warning_text = "Please enter some text"
    generating_text = "Generating response..."
    response_header = "<div style='direction: ltr;'><h3>Response:</h3></div>"
    evaluation_header = "<div style='direction: ltr;'><h3>Evaluation:</h3></div>"
    understood_text = "✅ Understood"
    not_understood_text = "❌ Not understood"

# Text area with appropriate direction
st.markdown(f"<div style='direction: {text_direction};'>", unsafe_allow_html=True)
user_input = st.text_area("", height=100, placeholder=placeholder_text)
st.markdown("</div>", unsafe_allow_html=True)

# Add a loading message while loading the model
loading_message = st.empty()
loading_message.info(loading_text)

# Set up environment for API key
setup_env_for_api_key()

# For large models, verify API key first
if ("llama" in model_option.lower() or "mistral" in model_option.lower() or "opt-1.3b" in model_option.lower() or "bloom" in model_option.lower()):
    if not has_api_key:
        api_msg = "API Key required for this model. Please enter a valid Hugging Face API Token." if not is_arabic else "مفتاح API مطلوب لهذا النموذج. الرجاء إدخال رمز API صالح من Hugging Face."
        loading_message.error(api_msg)
        
        # Show instructions on how to get an API key
        if is_arabic:
            st.markdown("""
            <div style='direction: rtl;'>
            <h4>كيفية الحصول على مفتاح API من Hugging Face:</h4>
            <ol>
                <li>قم بإنشاء حساب أو تسجيل الدخول إلى <a href='https://huggingface.co/' target='_blank'>Hugging Face</a></li>
                <li>انتقل إلى <a href='https://huggingface.co/settings/tokens' target='_blank'>صفحة إعدادات الرموز</a></li>
                <li>انقر على "New token" وأعط المفتاح اسماً واختر صلاحية "Read"</li>
                <li>انقر على "Generate token" وانسخ المفتاح</li>
                <li>الصق المفتاح في حقل إدخال مفتاح API أعلاه</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='direction: ltr;'>
            <h4>How to get a Hugging Face API Key:</h4>
            <ol>
                <li>Create an account or log in to <a href='https://huggingface.co/' target='_blank'>Hugging Face</a></li>
                <li>Go to <a href='https://huggingface.co/settings/tokens' target='_blank'>Token Settings Page</a></li>
                <li>Click "New token" and give it a name with "Read" permission</li>
                <li>Click "Generate token" and copy the token</li>
                <li>Paste the token in the API key input field above</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        
        st.stop()

# Try loading the model with simplified error handling
try:
    # Always pass API key if available, for both small and large models
    tokenizer, model = load_model(model_option, api_key)
    loading_message.empty()  # Clear the loading message when done
except Exception as e:
    error_msg = f"Error loading model: {str(e)}"
    loading_message.error(error_msg)
    st.error("Failed to load the selected model. Falling back to a smaller model (distilgpt2).")
    # Fallback to a reliable small model
    model_option = "distilgpt2"
    try:
        tokenizer, model = load_model(model_option, None)
        loading_message.success(f"Fallback to {model_option} successful!")
    except Exception as fallback_error:
        st.error(f"Fallback also failed: {str(fallback_error)}")
        st.stop()

# Add a submit button with clear styling
if st.button(button_text, key="generate_button", type="primary"):
    if not user_input.strip():
        st.warning(warning_text)
    else:
        with st.spinner(generating_text):
            try:
                # Format prompt based on model type
                if "llama" in model_option.lower():
                    # Llama-2 specific formatting
                    prompt = f"<s>[INST] {user_input} [/INST]"
                elif "mistral" in model_option.lower():
                    # Mistral specific formatting
                    prompt = f"<s>[INST] {user_input} [/INST]"
                elif "opt" in model_option.lower() or "bloom" in model_option.lower():
                    # OPT and BLOOM formatting (simple)
                    prompt = f"Question: {user_input}\nAnswer:"
                else:
                    # Generic formatting for other models
                    prompt = user_input
                
                # For all models
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # Adjust generation parameters based on model size
                generation_params = {
                    **inputs,
                    "do_sample": True,
                    "top_p": 0.9,
                    "temperature": 0.7,
                }
                
                if "llama" in model_option.lower() or "mistral" in model_option.lower() or "opt-1.3b" in model_option.lower() or "bloom" in model_option.lower():
                    generation_params["max_new_tokens"] = 256
                else:
                    generation_params["max_new_tokens"] = 150
                
                # Generate output
                with torch.no_grad():
                    outputs = model.generate(**generation_params)
                
                # Decode the response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the input prompt from the response if present
                if user_input in response:
                    response = response.replace(user_input, "").strip()
                
                # Simple evaluation algorithm
                word_count = len(response.split())
                has_relevant_content = word_count > 5
                
                # ChatGPT-like response box with better styling and proper text direction
                st.markdown(response_header, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 4px; background-color: #f7f7f8; border: 1px solid #e5e5e5; margin-bottom: 20px; direction: {text_direction};">
                    <span style="color: #000000; font-family: sans-serif; white-space: pre-wrap; display: block;">{response}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Evaluation result
                st.markdown(evaluation_header, unsafe_allow_html=True)
                if has_relevant_content:
                    st.markdown(f"<span style='color:green; font-size:18px; direction: {text_direction};'>{understood_text}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:red; font-size:18px; direction: {text_direction};'>{not_understood_text}</span>", unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Add footer with instructions
st.markdown("---")
if is_arabic:
    st.markdown("<div style='direction: rtl;'><h3>كيفية الاستخدام:</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='direction: rtl;'>
    1. اختر اللغة المفضلة في القائمة الجانبية
    2. أدخل مفتاح API من Hugging Face للوصول إلى النماذج الكبيرة
    3. اختر النموذج اللغوي من القائمة الجانبية
    4. أدخل سؤالك أو التعليمات في مربع النص
    5. انقر على زر التوليد
    6. شاهد استجابة النموذج مع تقييم ما إذا كان قد فهم المدخلات
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<div style='direction: ltr;'><h3>How to Use:</h3></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='direction: ltr;'>
    1. Choose your preferred language in the sidebar
    2. Enter your Hugging Face API key to access large models
    3. Select a language model from the sidebar
    4. Enter your question or instructions in the text box
    5. Click the generate button
    6. View the model's response with an evaluation of its understanding
    </div>
    """, unsafe_allow_html=True) 