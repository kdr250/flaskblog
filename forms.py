# Form
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError


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
