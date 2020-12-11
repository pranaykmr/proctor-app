import cv2
import os
import json
import threading
from datetime import datetime
from flask import Flask, Response, render_template, url_for
from proctorapp import app, db
from proctorapp.models import User, Session, Logs
from proctorapp.forms import RegistrationForm, LoginForm
from proctorapp.proctoring_models import BaseModel, EyeTracker, Mouth_Opening, Head_Position, Object_Detector
from flask import request

currFlag = True


@app.route("/")
@app.route("/index")
def home():
    global currFlag
    currFlag = False
    insertLogs()
    return render_template("index.html")


@app.route("/admin")
def admin_home():
    form = LoginForm()
    return render_template("indexAdmin.html", form=form)


@app.route("/sessionList.html")
def sessionList():
    data = getSessionData()
    return render_template(
        "sessionList.html",
        data={"user": {"name": "Pranay", "id": "prverma", "isAdmin": False}, "sessions": data},
    )


@app.route("/addStudent.html")
def addStudent():
    return render_template("addStudent.html", data={"user": {"name": "Pranay", "id": "prverma", "isAdmin": True}})


@app.route("/createSession.html")
def createSession():
    return render_template("createSession.html", data={"user": {"name": "Pranay", "id": "prverma", "isAdmin": True}})


@app.route("/dummy.html")
def dummy():
    return render_template("dummy.html")


@app.route("/sessionListAdmin.html")
def sessionListAdmin():
    data = getSessionData()
    return render_template(
        "sessionListAdmin.html",
        data={"user": {"name": "Pranay", "id": "prverma", "isAdmin": True}, "sessions": data},
    )


@app.route("/studentList.html")
def studentList():

    data = User.query.filter_by(isAdmin=False)

    return render_template(
        "studentList.html",
        data={"user": {"name": "Pranay", "id": "prverma", "isAdmin": True}, "students": data.all()},
    )


def getSessionData():
    return Session.query.all()


@app.route("/examPage.html")
def examPage():
    return render_template("examPage.html", data={"user": {"name": "Pranay", "id": "prverma", "isAdmin": False}})


@app.route("/run_model")
def run_model():
    video = cv2.VideoCapture(0)
    return Response(invoke_models(video), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/handle_data_add_student", methods=["GET", "POST"])
def handle_data_add_student():
    fname = request.form.get("fname", "")
    lname = request.form.get("lname", "")
    stdId = request.form.get("stdId", "")
    stdEmail = request.form.get("stdEmail", "")
    isAdmin = False

    try:
        user = User(username=f"{fname}_{lname}", email=stdEmail, password="asd", name=fname + lname, isAdmin=False)
        db.session.add(user)
        db.session.commit()
    except Exception as e:
        print(e)
    return studentList()


@app.route("/handle_data_create_session", methods=["GET", "POST"])
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
        os.remove("log.json")
        # with open("log.json", "w") as outfile:
        #     json.dump("", outfile)
    # global currFlag
    # if currFlag:
    #     timer()


# def timer():
#     threading.Timer(10.0, hello).start()