import os
from flask import Flask, render_template, request, jsonify
from inference import predict, initialize

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Edge Model on startup
try:
    initialize()
except Exception as e:
    print("Warning: ", e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def run_inference():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.jpg')
    file.save(filepath)
    
    try:
        results = predict(filepath)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from flask_cloudflared import run_with_cloudflared

if __name__ == '__main__':
    # Running locally with Cloudflare tunnel for public access
    run_with_cloudflared(app)
    app.run(host='0.0.0.0', port=5000, debug=False)
