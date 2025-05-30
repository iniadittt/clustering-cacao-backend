import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import io
import base64
from http import HTTPStatus
from dotenv import load_dotenv
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request
from flask_cors import CORS
from cacao import CacaoColorSegmentation

load_dotenv()

app = Flask(__name__)
CORS(app)
cacao = CacaoColorSegmentation("dummy_path", n_clusters=5)

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'storages/'
app.config['MODEL_KMEANS'] = './models/kmeans.joblib'

model_segmenter = cacao.load_model(app.config['MODEL_KMEANS'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
           
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'code': HTTPStatus.OK,
        'message': 'Success',
        'data': None,
    }), HTTPStatus.OK

@app.route('/predict', methods=['POST'])
def predictSegmenter():
    if model_segmenter:
        request_image = request.files.get('image')
        if request_image and allowed_file(request_image.filename):
            filename = secure_filename(request_image.filename)
            request_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(image_path):
                prediction, cluster_images, save_path = cacao.predict_and_visualize(image_path, model_segmenter)
                if prediction:
                    base64_image = None
                    with open(save_path, 'rb') as img_file:
                        base64_image = f'data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}'
                    return jsonify({
                        'status': {
                            'code': HTTPStatus.OK,
                            'message': 'OK',
                            'predicted': prediction['predicted_category'],
                            'confidence': prediction['confidence_score'],
                            'image_data': base64_image
                        },
                    }), HTTPStatus.OK
                else:
                    return jsonify({
                        'status': {
                            'code': HTTPStatus.INTERNAL_SERVER_ERROR,
                            'message': 'Prediction failed'
                        }
                    }), HTTPStatus.INTERNAL_SERVER_ERROR
            else:
                return jsonify({
                    'status': {
                        'code': HTTPStatus.BAD_REQUEST,
                        'message': 'Image not found'
                    }
                }), HTTPStatus.BAD_REQUEST
        else:
            return jsonify({
                'status': {
                    'code': HTTPStatus.BAD_REQUEST,
                    'message': 'Invalid or missing image file'
                }
            }), HTTPStatus.BAD_REQUEST
    else:
            return jsonify({
                'status': {
                    'code': HTTPStatus.INTERNAL_SERVER_ERROR,
                    'message': 'Model not loaded'
                }
            }), HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 9000)))