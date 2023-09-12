from flask import Flask, render_template, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
import torch
import easyocr

app = Flask(__name__)


def translate(text):
  model_name = "Helsinki-NLP/opus-mt-en-fr"
  tokenizer = MarianTokenizer.from_pretrained(model_name)
  model = MarianMTModel.from_pretrained(model_name)

  inputs = tokenizer(text, max_length=64, return_tensors="pt", truncation=True)

  with torch.no_grad():
      outputs = model.generate(**inputs)

  translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return translated_text


@app.route("/")
def home():
  return render_template('home.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
      request.files['file'].save('temp_file.png')

      reader = easyocr.Reader(['en'])

      result = reader.readtext('temp_file.png')

      text = ''
      for detection in result:
          text += f' {detection[1]}'
      
      translated_text = translate(text)

      return jsonify({"message": translated_text})
    except Exception as e:
      print(e)
      return jsonify({"message": "Erro na leitura do arquivo."})      


if __name__ == "__main__":
  app.run(port=8000, debug=True)