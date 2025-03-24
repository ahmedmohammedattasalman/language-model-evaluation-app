# Language Model Evaluation App 🤖

A Streamlit application for evaluating language models with support for Arabic and English. This app allows you to test and compare different language models, from small to medium-sized models.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ahmedmohammedattasalman/language-model-evaluation-app)

## Features 🌟

- 🌐 Bilingual Interface (Arabic/English)
- 🤖 Multiple Language Models Support
- 📊 Response Evaluation
- 🔄 Automatic Fallback System
- 🔑 API Key Management

## Models Available 🚀

### Small Models (No API Key Required)
- Facebook OPT-125M
- DistilGPT-2
- GPT-Neo-125M
- DialoGPT

### Medium Models (API Key Required)
- Facebook OPT-1.3B
- BLOOM-560M

## How to Use 📝

1. Choose your preferred language (Arabic/English)
2. Enter your Hugging Face API key (optional)
3. Select a language model
4. Enter your question or instructions
5. Click the generate button
6. View the model's response and evaluation

## API Key Setup 🔑

To use medium-sized models, you'll need a Hugging Face API key:

1. Create/login to your [Hugging Face account](https://huggingface.co/)
2. Go to [Token Settings](https://huggingface.co/settings/tokens)
3. Create a new token with "read" access
4. Copy and paste the token in the app

## Local Development 💻

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment 🚀

This app is deployed on Hugging Face Spaces. Visit the live demo [here](https://huggingface.co/spaces/ahmedmohammedattasalman/language-model-evaluation-app).

## License 📄

MIT License 