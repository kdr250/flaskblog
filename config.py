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

# PostgreSQL
db_name = os.environ['DBNAME']
db_host = os.environ['DBHOST']
db_user = os.environ['DBUSER']
db_password = os.environ['DBPASS']
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{db_user}:{db_password}@{db_host}/{db_name}'
db = SQLAlchemy(app)

# ローカル環境がMySQL、デプロイ先がHerokuの場合
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'mysql://root:@localhost/flaskblog'
# db = SQLAlchemy(app)

# ローカル環境がSQLite、デプロイ先がHerokuの場合
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///site.db'
# db = SQLAlchemy(app)

