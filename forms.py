# Form
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from models import User

class ResigtrationForm(FlaskForm):
  username = StringField('Username',
    validators=[DataRequired(), Length(min=2, max=20)])
  email = StringField('Email',
    validators=[DataRequired(), Email()])
  password = PasswordField('Password',
    validators=[DataRequired()])
  confirm_password = PasswordField('Confirmation',
    validators=[DataRequired(), EqualTo('password')])
  submit = SubmitField('Sign Up')

  def validate_username(self, username):
    user = User.query.filter_by(username=username.data).first()
    if user:
      raise ValidationError('入力されたユーザー名は他のユーザーによって既に使われています')

  def validate_email(self, email):
    email = User.query.filter_by(email=email.data).first()
    if email:
      raise ValidationError('入力されたEmailは他のユーザーによって既に使われています')

class LoginForm(FlaskForm):
  email = StringField('Email',
    validators=[DataRequired(), Email()])
  password = PasswordField('Password', validators=[DataRequired()])
  submit = SubmitField('Login')

class PostForm(FlaskForm):
  title = StringField('Title',
    validators=[DataRequired(), Length(min=1, max=100)])
  content = TextAreaField('Content',
    validators=[DataRequired()])
  submit = SubmitField('Post')
