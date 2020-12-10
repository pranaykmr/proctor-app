from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Date

app = Flask(__name__)
app.config["SECRET_KEY"] = "ce197258ca2656e13b284db6e6819d6e"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///procApp.db"
db = SQLAlchemy(app)

from proctorapp import routes