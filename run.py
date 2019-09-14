# Main
from flask import Flask, render_template, redirect, url_for, flash
from flask_login import LoginManager, current_user, login_user, logout_user
from flask_bcrypt import Bcrypt

# forms
from forms import ResigtrationForm, LoginForm

# DB
from flask_sqlalchemy import SQLAlchemy

# Config
app = Flask(__name__)
app.config['SECRET_KEY'] = 'd50afe8ef1fe6f6934245436e6a52776de99f8fa2b31e766991acaf6ef57'

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

db = SQLAlchemy(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/flaskblog'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

# models
from models import User, Post

# routes
@app.route('/')
def home():
  posts = [
    {
      'author': 'Taro',
      'title': 'First Post',
      'contet': 'The first post!',
      'date_posted': '2019/09/11',
    },
    {
      'author': 'Jiro',
      'title': 'Secnd Post',
      'contet': 'The second post!',
      'date_posted': '2019/09/13',
    }
  ]
  return render_template('home.html', posts=posts)
  # post = 'hello world'
  # return "Hello World"

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


if __name__ == '__main__':
  app.run(debug=True)

