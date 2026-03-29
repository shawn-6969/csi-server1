import numpy as np
from flask import Flask, request, jsonify
from scipy.signal import spectrogram
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "wifi_human_detector.h5")
model      = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded! Input shape: {model.input_shape}")

def process_csi(csi_buffer):
    """
    Replicates EXACTLY your real-time model pipeline:
    1. sample shape: (128, n_subcarriers)
    2. signal = mean of subcarriers 10:30 per packet
    3. spectrogram with fs=100
    4. log1p
    5. min-max normalize
    6. resize to 64x64
    7. reshape to (1, 64, 64, 1)
    """
    sample = np.array(csi_buffer, dtype=np.float32)  # (128, n_sub)

    # Step 1 — mean of subcarriers 10:30
    n_sub  = sample.shape[1]
    end    = min(30, n_sub)
    signal = np.mean(sample[:, 10:end], axis=1)       # (128,)

    # Step 2 — spectrogram
    f, t, Sxx = spectrogram(signal, fs=100)

    # Step 3 — log1p
    Sxx = np.log1p(Sxx)

    # Step 4 — min-max normalize
    Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min() + 1e-6)

    # Step 5 — resize to 64x64
    img = cv2.resize(Sxx, (64, 64))

    # Step 6 — reshape for model
    img = img.reshape(1, 64, 64, 1)

    return img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'csi' not in data:
            return jsonify({'error': 'No CSI data'}), 400

        # csi = list of amplitude arrays, shape (128, n_subcarriers)
        csi_buffer = data['csi']
        n_packets  = len(csi_buffer)

        print(f"Received {n_packets} packets, "
              f"{len(csi_buffer[0])} subcarriers each")

        if n_packets < 128:
            return jsonify({
                'error': f'Need 128 packets, got {n_packets}'
            }), 400

        # Process exactly like your real-time model
        img_input  = process_csi(csi_buffer[:128])

        # Predict
        prediction = model.predict(img_input, verbose=0)
        label_idx  = int(np.argmax(prediction))
        classes    = ["empty", "human_detected"]
        label      = classes[label_idx]

        prob_empty  = round(float(prediction[0][0]) * 100, 2)
        prob_human  = round(float(prediction[0][1]) * 100, 2)

        print(f"Result: {label} | "
              f"empty={prob_empty}% human={prob_human}%")

        return jsonify({
            'label'      : label,
            'confidence' : prob_human,
            'message'    : 'ALERT: Human detected!'
                           if label == 'human_detected'
                           else 'No human found',
            'prob_empty' : prob_empty,
            'prob_human' : prob_human
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status' : 'ok',
        'model'  : str(model.input_shape),
        'classes': ['empty', 'human']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)