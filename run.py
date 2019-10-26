# Main
from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

import io
from janome.tokenizer import Tokenizer
import pickle
from flask import render_template, redirect, url_for, flash, request, json, jsonify
from config import app, db, login_manager, bcrypt
from forms import ResigtrationForm, LoginForm, PostForm
from models import User, Post
from flask_login import current_user, login_user, logout_user, login_required

# Bot
import tensorflow as tf
import keras
from keras.backend import tensorflow_backend as backend
from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random

# モデルや辞書・リストを先に読み込み
model=load_model('harrybotter/harry_wakati.h5')
model._make_predict_function()
graph = tf.compat.v1.get_default_graph()
with open("harrybotter/wakati_harry.picle", mode="rb") as f:
      wakati_data = pickle.load(f)

maxlen = 20
step = 1
chars = sorted(list(set(wakati_data)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
next_chars = []
for i in range(0, len(wakati_data) - maxlen, step):
    next_chars.append(wakati_data[i + maxlen])

bot_user_id = User.query.filter_by(username="ChatBotter").first().id


# モデルを実行するための関数
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def bot_run(input_words=""):
    global char_indices
    global indices_char
    global next_chars
    
    print()
    print('----- Generating word after Epoch:')

    tokenizer = Tokenizer()

    start_index = random.randint(0, len(wakati_data) - maxlen - 1)

    for diversity in [0.2]:  # diversityは今回、一つの値で実施
        print('----- diversity:', diversity)

        generated = ''
        input_words = input_words
        input_word = tokenizer.tokenize(input_words, wakati=True)
        input_word = list(val_word for val_word in input_word if val_word in wakati_data)
        if len(input_word) > 20:
            input_word = input_word[-20:]

        sentence = wakati_data[start_index:start_index+maxlen]
        generated += ''.join(input_word)
        print('----- Generating with seed: "' + ''.join(input_word) + '"')

        for i in range(80):
            x_pred = np.zeros((1, maxlen, len(chars)))
            if i == 0:
                arg_str = input_word
            else:
                arg_str = sentence
            for t, char in enumerate(arg_str):
                x_pred[0, t, char_indices[char]] = 1.

            global graph
            with graph.as_default():
                preds = model.predict(x_pred, verbose=0)[0]
                
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:]
            sentence.append(next_char)

            if (next_char == "！") or (next_char == "？") or (next_char == "。"):
                break

    return generated


# routes
@app.route('/')
def index():
  return render_template('index.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
  posts = Post.query.all()
  form = PostForm()
  if form.validate_on_submit():
    post = Post(title=form.title.data, content=form.content.data, user_id=current_user.id)
    db.session.add(post)
    db.session.commit()
    flash(f'Post Success!', 'success')
    return redirect(url_for('home'))
  return render_template('home.html', posts=posts, form=form)

@app.route('/post_ajax', methods=['POST'])
def post_ajax():
  title = request.form['title']
  content = request.form['content']
  post = Post(title=title, content=content, user_id=current_user.id)
  db.session.add(post)
  db.session.commit()
  same_author = 0
  if post.author == current_user:
      same_author = 1
  # Sending content to Bot
  bot_return = bot_run(content)
  bot_title = f"Re: {title}"
  global bot_user_id
  bot_post = Post(title=bot_title, content=bot_return, user_id=bot_user_id)
  db.session.add(bot_post)
  db.session.commit()
  return jsonify({'id': post.id , 'title': title, 'content': content,
                  'date_posted': post.date_posted.strftime('%Y年%m月%d日'),
                  'authorname': post.author.username, 'same': same_author})

@app.route('/post_api', methods=['POST'])
def post_api():
  last_post_id = int(request.json["id"])
  posts = db.session.query(Post).filter(Post.id > last_post_id).all()
  list_json = []
  same_author = 0
  for post in posts:
    if post.author == current_user:
      same_author = 1
    dict_json = {'id': post.id, 'title': post.title,
      'content': post.content, 'date_posted': post.date_posted.strftime('%Y年%m月%d日'),
      'authorname': post.author.username, 'same': same_author}
    list_json.append(dict_json)
  json = jsonify(list_json)
  return json

@app.route('/about')
def about():
  title = "About"
  return render_template('about.html', title=title)

@app.route('/register', methods=['GET', 'POST'])
def register():
  if current_user.is_authenticated:
    return redirect(url_for('home'))
  form = ResigtrationForm()
  if form.validate_on_submit():
    hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
    user = User(username=form.username.data, email=form.email.data, password=hashed_password)
    db.session.add(user)
    db.session.commit()
    flash(f'Account Created! Now Login!', 'success')
    return redirect(url_for('login'))
  return render_template('register.html', form=form, title='Register')

@app.route('/login', methods=['GET', 'POST'])
def login():
  if current_user.is_authenticated:
    return redirect(url_for('home'))
  form = LoginForm()
  password_incorrect = 0
  if form.validate_on_submit():
    user = User.query.filter_by(email=form.email.data).first()
    if user and bcrypt.check_password_hash(user.password, form.password.data):
      login_user(user)
      flash(f'You Login!', 'success')
      return redirect(url_for('home'))
    else:
      password_incorrect = 1
  return render_template('login.html', form=form, title='Login', password_incorrect=password_incorrect)

@app.route('/logout')
def logout():
  logout_user()
  flash(f'You Logout!', 'danger')
  return redirect(url_for('index'))

@app.route('/post/new', methods=['GET', 'POST'])
@login_required
def new_post():
  form = PostForm()
  if form.validate_on_submit():
    post = Post(title=form.title.data, content=form.content.data, user_id=current_user.id)
    db.session.add(post)
    db.session.commit()
    flash(f'Post Success!', 'success')
    return redirect(url_for('home'))
  return render_template('post.html', title='New Post', form=form)


if __name__ == '__main__':
  # app.run(debug=True)
  app.run(debug=False)
