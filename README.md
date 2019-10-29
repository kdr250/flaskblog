ChatBotter
====
チャットボットと会話できるWebアプリです。  
ChatBotterへのリンクは[こちらから](http://chatbotter.azurewebsites.net/)。


## 概要
ChatBotterは、ユーザーの投稿に対してボットが自動で応答するチャットアプリです。
今回、Python製WebフレームワークのFlaskを用いてWebアプリ開発を行いました。
必要とされる機能を一から開発することによってどういう実装でその機能が成り立っているのか、Webアプリに基本的に備わっている機能を実装する力などが身に付きました。
また、PaaSにデプロイし、デモサイトを構築することを考えていたため、ユーザーが分かりやすいようなUI画面を意識し制作しました。

## 主な機能
主な機能は以下のとおりです。
1. ユーザー登録機能
2. ログイン・ログアウト機能
3. チャット投稿機能
4. ボット応答機能

## データベース設計
ER図は下記の通りです。  
<img src="https://i.gyazo.com/5c4595591fc0566e0fb6edd188c6b3bd.png" alt="Image from Gyazo" width="757"/>

## 開発環境
### ハードウェア・OS  
* MacBook Pro  
* OS: MacOS Mojave (10.14.6)  
* CPU: Intel Core i5 2.4 GHz  
* メモリ: 8GB 

### 使用言語・データベース
* Python (3.7.4)
* HTML
* CSS
* JavaScript
* PostgreSQL

### 使用ツール・ライブラリ
* Flask
* jQuery
* Visual Studio Code
* Bootstrap4

## インストール
ローカル環境へのインストールは、ターミナルにて下記コマンドを実行ください。
またあらかじめ、PostgreSQLなどのDBに当アプリ用のデータベース・接続用ユーザーをご用意ください。初期状態ではPostgreSQL用の設定になっています。
```bash:
$ git clone https://github.com/kdr250/flaskblog
$ cd flaskblog
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ export DBHOST="localhost"
$ export DBNAME="作成したデータベース名"
$ export DBUSER="データベース接続用ユーザー名"
$ export DBPASS="データベース接続用パスワード"
$ export CHATBOTTER_SECRET_KEY="任意の文字列"
$ source ~/.bash_profile
```
Python対話シェルより下記コマンドを実行し、テーブル作成とボット用ユーザー登録を行ってください。
```python:
$ python
from config import db
from models import User, Post
db.create_all()
db.session.commit()
from config import bcrypt
hashed_password = bcrypt.generate_password_hash("任意の文字列").decode("utf-8")
bot = User(username="ChatBotter", email="chatbotter@example.com", password=hashed_password)
db.session.add(bot)
db.session.commit()
```

## 起動
下記コマンドで起動します。
```bash:
$ python app.py       # http://localhost:5000で接続
または
$ gunicorn app:app    # http://localhost:8000で接続
```
