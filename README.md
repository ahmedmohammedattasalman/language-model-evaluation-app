# Language Model Evaluation App

A Streamlit application for evaluating language models with support for Arabic and English.

## Features

- Supports both Arabic and English interfaces
- Evaluates responses from various language models
- Includes small models that run without an API key
- Optional support for medium-sized models with Hugging Face API key
- Automatic fallback to smaller models if larger ones fail to load

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   streamlit run app.py
   ```

## Models

The app includes the following models:

### Small Models (No API key required)
- Facebook OPT-125M
- DistilGPT-2
- GPT-Neo-125M
- DialoGPT

### Medium Models (Hugging Face API key recommended)
- Facebook OPT-1.3B
- BLOOM-560M

## How to Get a Hugging Face API Key

1. Create an account or log in to [Hugging Face](https://huggingface.co/)
2. Go to the [Token Settings Page](https://huggingface.co/settings/tokens)
3. Click "New token" and give it a name with "Read" permission
4. Click "Generate token" and copy the token
5. Paste the token in the API key input field in the app

## Usage

1. Choose your preferred language (Arabic/English)
2. Enter your Hugging Face API key (optional)
3. Select a language model
4. Enter your question or instructions
5. Click the generate button
6. View the model's response and evaluation 