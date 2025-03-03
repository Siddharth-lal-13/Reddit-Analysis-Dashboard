from flask import Flask
from src.main import app as dash_app
import os

flask_app = Flask(__name__)
dash_app.server = flask_app  # Attach Dash to Flask

if __name__ == "__main__":
    flask_app.run(host='0.0.0.0', port=int(os.getenv('PORT', 10000)))
    