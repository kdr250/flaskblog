# Main
from flask import Flask, render_template, redirect, url_for, flash, request, json, jsonify
from flask_login import LoginManager, current_user, login_user, logout_user, login_required
from flask_bcrypt import Bcrypt
# from flask_bootstrap import Bootstrap

# forms
from forms import ResigtrationForm, LoginForm, PostForm

# DB
from flask_sqlalchemy import SQLAlchemy

# Config
app = Flask(__name__)
app.config['SECRET_KEY'] = 'd50afe8ef1fe6f6934245436e6a52776de99f8fa2b31e766991acaf6ef57'

# bootstrap = Bootstrap(app)

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

db = SQLAlchemy(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/flaskblog'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

# models
from models import User, Post

# routes
@app.route('/', methods=['GET', 'POST'])
def home():
  # posts = Post.query.order_by(Post.date_posted.desc())
  posts = Post.query.all()
  form = PostForm()
  if form.validate_on_submit():
    post = Post(title=form.title.data, content=form.content.data, user_id=current_user.id)
    db.session.add(post)
    db.session.commit()
    flash(f'Post Success!', 'success')
    return redirect(url_for('home'))
  return render_template('home.html', posts=posts, form=form)
  # post = 'hello world'
  # return "Hello World"

@app.route('/post_ajax', methods=['POST'])
def post_ajax():
  title = request.form['title']
  content = request.form['content']
  post = Post(title=title, content=content, user_id=current_user.id)
  db.session.add(post)
  db.session.commit()
  return jsonify({'id': post.id , 'title': title, 'content': content, 'date_posted': post.date_posted.strftime('%Y年%m月%d日'), 'authorname': post.author.username})
  # name = request.form['name']
  # return jsonify({'result': 'ok', 'value': name})

@app.route('/post_api', methods=['POST'])
def post_api():
  last_post_id = int(request.json["id"])
  print(last_post_id)
  posts = Post.query.filter(Post.id > last_post_id).all()
  # print(posts)    # [Post('aaa', '2019-09-15 04:14:12')]
  list_json = []
  for post in posts:
    dict_json = {'id': post.id, 'title': post.title,
      'content': post.content, 'date_posted': post.date_posted.strftime('%Y年%m月%d日'),
      'authorname': post.author.username}
    list_json.append(dict_json)
  print(list_json)
  json = jsonify(list_json)
  return json

@app.route('/about')
def about():
  post = "About Page"
  title = "About"
  return render_template('about.html', post=post, title=title)

@app.route('/register', methods=['GET', 'POST'])
def register():
  if current_user.is_authenticated:
    return redirect(url_for('home'))
  form = ResigtrationForm()
  if form.validate_on_submit():
    # password = form.password.data
    hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
    user = User(username=form.username.data, email=form.email.data, password=hashed_password)
    db.session.add(user)
    db.session.commit()
    flash(f'Account Created! Now Login!', 'success')
    return redirect(url_for('home'))
  return render_template('register.html', form=form, title='Register')

@app.route('/login', methods=['GET', 'POST'])
def login():
  if current_user.is_authenticated:
    return redirect(url_for('home'))
  form = LoginForm()
  if form.validate_on_submit():
    user = User.query.filter_by(email=form.email.data).first()
    if user and bcrypt.check_password_hash(user.password, form.password.data):
      login_user(user)
      flash(f'You Login!', 'success')
      return redirect(url_for('home'))
  return render_template('login.html', form=form, title='Login')

@app.route('/logout')
def logout():
  logout_user()
  flash(f'You Logout!', 'danger')
  return redirect(url_for('home'))

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

# Ajaxテスト
@app.route('/hello')
def hello_view():
  return render_template('hello.html', title='random_greeting')

# ランダムに投稿を取得してjsonとして返す
@app.route('/greeting_post', methods=['POST'])
def greeting_process():
  post = Post.query.get(request.json["key"])
  greeting = post.content
  print(greeting)

  return_json = {
    'greeting': greeting,
  }
  return jsonify(ResultSet=json.dumps(return_json))


if __name__ == '__main__':
  app.run(debug=True)

