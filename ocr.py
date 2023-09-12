import easyocr


reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext('./resources/test.png')

print(result)