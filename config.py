# Main
from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt


# Config
app = Flask(__name__)
app.config['SECRET_KEY'] = 'd50afe8ef1fe6f6934245436e6a52776de99f8fa2b31e766991acaf6ef57'

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

db = SQLAlchemy(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/flaskblog'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

# # HarryBotter
# import tensorflow as tf
# # import keras
# from keras.models import load_model
# import pickle

# Bot
# keras.backend.clear_session()
# graph = tf.get_default_graph()
# model=load_model('flaskblog/harrybotter/harry_wakati.h5')
# with open("flaskblog/harrybotter/wakati_harry.picle", mode="rb") as f:
#     wakati_data = pickle.load(f)

# keras.backend.clear_session()
# graph = tf.get_default_graph()

# from flaskblog import routes


