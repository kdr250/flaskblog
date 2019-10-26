# Main
from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import os

# Config
app = Flask(__name__)
app.config['SECRET_KEY'] = 'd50afe8ef1fe6f6934245436e6a52776de99f8fa2b31e766991acaf6ef57'

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

db = SQLAlchemy(app)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'mysql://root:@localhost/flaskblog'

# SQLiteの場合は下記
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///site.db'

