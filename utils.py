import os
from openai import OpenAI
import re

def check_api():
    client = OpenAI(
        base_url='YOUR_API', 
        api_key='YOUR_KEY'
        )
    try:
        models = client.models.list()
        print("API Key Available! Models:", [model.id for model in models.data])
    except Exception as e:
        print(f"API Key not Available: {str(e)}")

def extract_option(s):
    g = re.search('[A-Za-z]', s)
    option = g.group()

    return option.upper()

def count_token_usage(total, current):
    for k in total.keys():
        total[k] += current[k]
    
    return total
    
def has_nonsense(text):
    for char in text:
        if not(char.isascii() or 0x4E00 <= ord(char) <= 0x9FFF):
            return True
    return False