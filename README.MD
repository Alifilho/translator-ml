# Translator ML

## Text Translation Micro Application

This Python-based micro application utilizes Flask to create a server that serves an HTML page for user interaction. The application's main functionality is to receive an image containing text, read the text using the EasyOCR library, and then translate it from English to French using the Transformers library with pre-trained models. The translated text is then sent back to the frontend and displayed on the page.

## Prerequisites

Before running the application, ensure you have the following prerequisites installed:

- Python
- Flask library
- EasyOCR library
- Transformers library

You can install the required libraries using pip:

```
   pip install Flask easyocr transformers
```

## Running the Application

To start the application, you can use the provided Makefile with the following command:

```
   make run
```

This command will handle the necessary steps to launch the application.

## How the Application Works

1. User Interaction: The HTML page served by Flask contains a file input field. Users can upload an image containing text.

2. Backend Processing:
   - The backend (Python) uses the EasyOCR library to extract the text from the uploaded image.
   - The extracted text is then passed to the Transformers library to perform the translation from English to French using machine learning.

3. Displaying the Result:
   - The translated text is sent back to the frontend and displayed on the page.

## Folder Structure

- /templates: Contains the HTML file for user interaction.
- /resources: sample files to test
- /tmp: folders for storing temporary files
- app.py: The main Python file that handles server creation and backend processing.

## Additional Notes

Feel free to customize and expand upon this micro application according to your project's specific requirements. You can also incorporate additional features or optimize the existing codebase for further functionality. Don't forget to document any changes you make for future reference.

If you encounter any issues or have questions, feel free to reach out. Happy coding!
