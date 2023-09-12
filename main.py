from flask import Flask, render_template, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
from os.path import join
from os import getcwd, remove
from werkzeug.utils import secure_filename
from torch import no_grad
from easyocr import Reader

app = Flask(__name__)


def translate(text):
  model_name = "Helsinki-NLP/opus-mt-en-fr"
  tokenizer = MarianTokenizer.from_pretrained(model_name)
  model = MarianMTModel.from_pretrained(model_name)

  inputs = tokenizer(text, max_length=64, return_tensors="pt", truncation=True)

  with no_grad():
      outputs = model.generate(**inputs)

  translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return translated_text


@app.route("/")
def home():
  return render_template('home.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
      temp_dir = join(getcwd(), 'tmp')

      filename = secure_filename(request.files['file'].filename)

      request.files['file'].save(join(temp_dir, filename))

      reader = Reader(['en'])

      result = reader.readtext(join(temp_dir, filename))

      text = ''
      for detection in result:
          text += f' {detection[1]}'
      
      translated_text = translate(text)

      remove(join(temp_dir, filename))

      return jsonify({"message": translated_text})
    except Exception as e:
      print(e)
      return jsonify({"message": "Erro na leitura do arquivo."})      


if __name__ == "__main__":
  app.run(port=8000, debug=True)