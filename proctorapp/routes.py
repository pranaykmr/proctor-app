from flask import render_template, url_for
from proctorapp import app
from proctorapp.forms import RegistrationForm, LoginForm

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/admin')
def admin_home():
    form = LoginForm()
    return render_template('indexAdmin.html', form=form)



@app.route('/sessionList.html')
def sessionList():
    return render_template('sessionList.html')


@app.route('/addStudent.html')
def addStudent():
    return render_template('addStudent.html')


@app.route('/createSession.html')
def createSession():
    return render_template('createSession.html')


@app.route('/dummy.html')
def dummy():
    return render_template('dummy.html')


@app.route('/sessionListAdmin.html')
def sessionListAdmin():
    return render_template('sessionListAdmin.html')


@app.route('/studentList.html')
def studentList():
    return render_template('studentList.html')


@app.route('/examPage.html')
def examPage():
    return render_template('examPage.html')