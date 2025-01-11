from flask import Flask, jsonify, request
from helper import predict
from io import BytesIO

app = Flask(__name__)

@app.route('/predict/', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if not file:
                return jsonify({"error": "No file uploaded."}), 400
            img_bytes = file.read()
            pred_label = predict(BytesIO(img_bytes))
            return jsonify({"label": f"{pred_label}"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)