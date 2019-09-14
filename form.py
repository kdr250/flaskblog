# Form
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, BooleanField
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
