from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def home():
  post = "Home Page"
  return render_template('home.html', post=post)
  # return "Hello World"

@app.route('/about')
def about():
  post = "About Page"
  title = "About"
  return render_template('about.html', post=post, title=title)

if __name__ == '__main__':
  app.run()

