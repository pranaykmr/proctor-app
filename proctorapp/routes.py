import cv2
import os
import json
import threading
from datetime import datetime
from flask import Flask, Response, render_template, url_for, redirect, flash
from proctorapp import app, db
from proctorapp.models import User, Session, Logs
from proctorapp.proctoring_models import BaseModel, EyeTracker, Mouth_Opening, Head_Position, Object_Detector
from flask import request

currFlag = True
session = {}


@app.route("/")
@app.route("/index")
def home():
    return render_home(isAdmin=False)


@app.route("/admin")
def admin_home():
    return render_home(isAdmin=True)


def render_home(isAdmin=False):
    global currFlag
    currFlag = False
    insertLogs()
    global session
    if "user" in session and not session["user"].isAdmin:
        return sessionList()
    elif "user" in session and session["user"].isAdmin:
        return sessionListAdmin()
    else:
        session = {}
        if isAdmin:
            return render_template("indexAdmin.html")
        else:
            return render_template("index.html")


@app.route("/logout")
def logout():
    global currFlag
    currFlag = False
    insertLogs()
    global session
    session = {}
    return render_template("index.html")


@app.route("/sessionList.html", methods=["POST"])
def sessionList():
    if "user" in session:
        uname = session["user"].username
        passwd = session["user"].password
    else:
        uname = request.form.get("uname")
        passwd = request.form.get("pass")

    user = User.query.filter_by(username=uname, password=passwd, isAdmin=False).first()

    if user:
        data = getSessionData()
        session["user"] = user
        return render_template("sessionList.html", data={"user": session["user"], "sessions": data})
    flash('Login Unsuccessful. Please check username and password', 'danger')
    return redirect(url_for("home"))


@app.route("/addUser.html")
def addUser():
    return render_template("addUser.html", data={"user": session["user"]})


@app.route("/createSession.html")
def createSession():
    return render_template("createSession.html", data={"user": session["user"]})


@app.route("/dummy.html")
def dummy():
    return render_template("dummy.html")


@app.route("/showUserData.html", methods=["GET"])
def showUserData():
    stdId = request.args.get("studentId")
    user = User.query.filter_by(id=stdId).first()
    return render_template("showUserData.html", data={"user": session["user"], "student": user})


@app.route("/sessionListAdmin.html", methods=["GET", "POST"])
def sessionListAdmin():
    if "user" in session:
        uname = session["user"].username
        passwd = session["user"].password
    else:
        uname = request.form.get("uname")
        passwd = request.form.get("pass")

    user = User.query.filter_by(username=uname, password=passwd, isAdmin=True).first()

    if user:
        session["user"] = user
        data = getSessionData()
        return render_template(
            "sessionListAdmin.html",
            data={"user": session["user"], "sessions": data},
        )
    flash('Login Unsuccessful. Please check username and password', 'danger')
    return redirect(url_for("admin_home"))


@app.route("/studentList.html")
def studentList():

    data = User.query.filter_by(isAdmin=False).all()
    # TODO check if data is not empty
    return render_template(
        "studentList.html",
        data={"user": session['user'], "students": data},
    )


def getSessionData():
    return Session.query.all()


@app.route("/examPage.html")
def examPage():
    return render_template("examPage.html", data={"user": session["user"]})


@app.route("/run_model")
def run_model():
    video = cv2.VideoCapture(0)
    return Response(invoke_models(video), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/handle_data_add_user", methods=["POST"])
def handle_data_add_user():
    fname = request.form.get("fname", "")
    lname = request.form.get("lname", "")
    userId = request.form.get("userId", "")
    userEmail = request.form.get("userEmail", "")
    isAdmin = True if request.form.get("isAdmin", False) == "on" else False

    try:
        user = User(username=f"{userId}", email=userEmail, password="asd", name=f"{fname} {lname}", isAdmin=isAdmin)
        db.session.add(user)
        db.session.commit()
    except Exception as e:
        print(e)
        # send error message on UI
    return studentList()


@app.route("/handle_data_create_session", methods=["POST"])
def handle_data_create_session():
    strtDate = datetime.strptime(request.form.get("strtDate", ""), "%Y-%m-%dT%H:%M")
    endDate = datetime.strptime(request.form.get("endDate", ""), "%Y-%m-%dT%H:%M")
    sessName = request.form.get("sessionName", "")
    sessionNotes = request.form.get("sessionNotes", "")
    try:
        session = Session(sessionname=sessName, startdate=strtDate, enddate=endDate, sessionnotes=sessionNotes)
        db.session.add(session)
        db.session.commit()
    except Exception as e:
        print(e)

    return sessionListAdmin()


def invoke_models(video):
    try:
        global currFlag
        currFlag = True
        eye_tracker = EyeTracker(video)
        mouth_open = Mouth_Opening(video)
        head_pos = Head_Position(video)
        object_detector = Object_Detector(video)

        logger = {"head_logger": [], "mouth_logger": [], "phone_logger": [], "eye_logger": []}
        # timer()

        while currFlag:
            logger["head_logger"].extend(head_pos.head_position())
            logger["mouth_logger"].extend(mouth_open.mouth_opening())
            logger["eye_logger"].extend(eye_tracker.eye_detector())
            logger["phone_logger"].extend(object_detector.person_and_phone())
            # json_object=json.dumps(logger, indent = 4)
            with open("log.json", "w") as outfile:
                json.dump(logger, outfile)
    except Exception as e:
        print(e)


def insertLogs():
    try:
        data = None
        with open("log.json") as f:
            data = json.loads(f.read())
    except Exception as e:
        print(e)
    if data:
        try:
            logs = Logs(rawData=str.encode(json.dumps(data)), userId=1, timestamp=datetime.now())
            db.session.add(logs)
            db.session.commit()
        except Exception as e:
            print(e)
        # os.remove("log.json")
        # with open("log.json", "w") as outfile:
        #     json.dump("", outfile)
    # global currFlag
    # if currFlag:
    #     timer()


# def timer():
#     threading.Timer(10.0, hello).start()