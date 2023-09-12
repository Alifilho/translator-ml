import torch
from transformers import MarianMTModel, MarianTokenizer

def translate(text):
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    inputs = tokenizer(text, max_length=64, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(**inputs)

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

input_text = "My name is Sarah and I live in London"
translated_text = translate(input_text)
print(f"Input: {input_text}")
print(f"Translated: {translated_text}")
