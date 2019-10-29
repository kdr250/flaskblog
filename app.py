# Main
from flask import Flask, Response, render_template, redirect, url_for, flash, request, json, jsonify
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

import io
import pickle
import threading

from config import app, db, login_manager, bcrypt
from forms import ResigtrationForm, LoginForm, PostForm
from models import User, Post
from flask_login import current_user, login_user, logout_user, login_required

# ChatBot
import math
import tensorflow as tf

from chatbot.chat_answer import Dialog, load_data, initialize_models, encode_request, generate_response

# ChatBot用データロード、初期設定等
word_indices ,indices_word ,words ,maxlen_e, maxlen_d ,freq_indices = load_data()
vec_dim = 400
n_hidden = int(vec_dim*1.5 )    # 隠れ層の次元

# ChatBot用入出力次元
input_dim = len(words)
output_dim = math.ceil(len(words) / 8)
graph = tf.compat.v1.get_default_graph()

#モデル初期化
model, encoder_model ,decoder_model = initialize_models('param_001' ,maxlen_e, maxlen_d,
                                                      vec_dim, input_dim, output_dim, n_hidden)


# ChatBotのメイン処理
def bot_run(input_words):
    global word_indices
    global words
    global indices_word
    global maxlen_e
    global maxlen_d
    global freq_indices
    global n_hidden
    global output_dim
    global encoder_model
    global decoder_model
    global graph

    # 入力文の品詞分解とインデックス化 
    e_input = encode_request(input_words, maxlen_e, word_indices, words, encoder_model)

    with graph.as_default():
      # 応答文組み立て
      decoded_sentence = generate_response(e_input, n_hidden, maxlen_d, output_dim, word_indices, 
                                              freq_indices, indices_word, encoder_model, decoder_model)

    return decoded_sentence

# 並列処理
class MyThread(threading.Thread):
    def __init__(self, title, content):
        super(MyThread, self).__init__()
        self.stop_event = threading.Event()
        self.title = title
        self.content = content

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
          # ChatBot用ユーザーidセット
          bot_user_id = User.query.filter_by(username="ChatBotter").first().id
          bot_return = bot_run(self.content)
          bot_title = f"Re: {self.title}"
          bot_post = Post(title=bot_title, content=bot_return, user_id=bot_user_id)
          db.session.add(bot_post)
          db.session.commit()

        finally:
            print('heavy process is finished\n')

jobs = {}


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
  same_author = 1

  t = MyThread(title, content)
  t.start()
  jobs[id] = t

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
