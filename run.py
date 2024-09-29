from flask import Flask
from flask_cors import CORS
from config import config
from app.routes.object_detection import object_detection
from app.routes.image_processing import image_processing
from app.routes.object_segmentation import object_segmentation
from waitress import serve
import os

def create_app(config_name='development'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    CORS(app)

    # Register blueprints
    app.register_blueprint(object_detection)
    app.register_blueprint(image_processing)
    app.register_blueprint(object_segmentation)

    return app

if __name__ == '__main__':
    app = create_app(os.getenv('FLASK_ENV', 'development'))
    if app.config['DEBUG']:
        app.run(host='0.0.0.0', port=5000)
    else:
        serve(app, host='0.0.0.0', port=5000)