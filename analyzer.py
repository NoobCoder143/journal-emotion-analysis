from transformers import pipeline
import google.generativeai as genai
from dotenv import load_dotenv
import os

# loading the api key from .env file
load_dotenv(dotenv_path='.env.local')
api_key=os.getenv("api_key")

#the user input
user_input="I really scared beacause i hava an interview tomorrow"

def analyze(user_input):
    #Using the predefined model from HuggingFace for emotion detection
    classifier=pipeline(task="text-classification",model="SamLowe/roberta-base-go_emotions")
    result=classifier(user_input)
    emotion=result[0]['label']

    # using the api key loaded from .env to access the gemini model
    genai.configure(api_key=api_key)
    model=genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    #this is the prompt given to the gemini model
    response=model.generate_content(f"""
    You are a caring and supportive friend. Based on the following input and detected emotion, write a short, friendly mental health tip. Keep it warm, understanding, and natural,Avoid sounding robotic or overly formal.

    User Input:{user_input}
    Detected Emotion: {emotion}

    Tip: 
    """
    )
    #this is the output
    print(f"User Prompt : {user_input}")
    print(f"Detected emotion : {emotion}")
    print(f"Generated tip : {response.text}")