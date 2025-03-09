from flask import Flask, request, jsonify
from flask_cors import CORS  # Importar CORS
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import MarianMTModel, MarianTokenizer
import torch
from threading import Thread


# Crear la app de Flask
app = Flask(__name__)

# Habilitar CORS para todas las rutas
CORS(app)  # Esto habilita CORS para todos los orI­genes

# Cargar el modelo 1
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Cargar el modelo 2
model_name_helsinki = "Helsinki-NLP/opus-mt-en-es"  #   Inglés - Español
tokenizer_helsinki = MarianTokenizer.from_pretrained(model_name_helsinki)
model_helsinki = MarianMTModel.from_pretrained(model_name_helsinki)



# HTML como un string (lo puedes modificar)
html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prueba de Servicio de Traduccion</title>
    <script>
        async function sendPrediction() {
            const text = document.getElementById("inputText").value;
            const response = await fetch("http://localhost:5555/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();
            document.getElementById("predictionResult").innerText = `Predicción: ${data.prediction}`;
        }
    </script>
</head>
<body>
    <div style="max-width: 600px; margin: 0 auto; text-align: center;">
        <h1>Predicción de Traduccion ENG - ESP</h1>
        <textarea id="inputText" rows="4" cols="50" placeholder="Write something in english..."></textarea>
        <br><br>
        <button onclick="sendPrediction()">Obtener Predicción</button>
        <p id="predictionResult" style="margin-top: 20px; font-size: 18px; font-weight: bold;"></p>
    </div>
</body>
</html>
"""

# Ruta para servir el HTML
@app.route("/")
def home():
    return html_content

# Ruta para la prediccion
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    inputs_helsinki = tokenizer_helsinki(data["text"], return_tensors="pt", truncation=True, padding=True)
    inputs=tokenizer(data["text"], return_tensors="pt")
    with torch.no_grad():
        outputs_helsinki = model_helsinki.generate(**inputs_helsinki)
        prediction_helsinki=tokenizer_helsinki.batch_decode(outputs_helsinki, skip_special_tokens=True)

        outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("es"))
        prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        final_text=prediction_helsinki+prediction
    return jsonify({"prediction":final_text})

# Ejecutar Flask en un hilo
def run_flask():
    app.run(host="0.0.0.0", port=5555)

# Crear un hilo para ejecutar Flask
thread = Thread(target=run_flask)
thread.start()