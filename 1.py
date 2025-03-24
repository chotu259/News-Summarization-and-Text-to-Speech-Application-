import json
import gradio as gr
from playsound import playsound
from gtts import gTTS
from googletrans import Translator
from transformers import pipeline
import main  # Ensure this module exists with the correct functions
import asyncio
loop = asyncio.get_event_loop()
translator = Translator()
# Load summarization and text generation models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
comparison_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")

def text_to_speech(company_name):
    try:
        # Fetch data
        data = main.get_data(company_name)
        json_data = json.loads(data)
        # Extract and summarize news
        company_news, text_output = main.extract_news(company_name, json_data, summarizer, comparison_pipeline)
        print(f"Company News: {company_news}\n\n")
        print(f"Summarized Text: {text_output}")

        # Translate text
        translated_text = loop.run_until_complete(translator.translate(text_output, src='en', dest='hi'))
        print(translated_text.text)  # Correct output
        tts = gTTS(translated_text.text, lang='hi')
        audio_file = "translated_speech.mp3"
        tts.save(audio_file)
        return audio_file
        # Convert to speech

        #return audio_file  # Gradio will handle playback
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=text_to_speech,
    inputs=gr.Textbox(placeholder="Enter Company name"),
    outputs="audio",
    title="English to Hindi Speech Converter",
    description="Enter English text, and it will be translated into Hindi and spoken aloud."
)

iface.launch()
