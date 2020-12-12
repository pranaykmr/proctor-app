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


'''
Login routes for Admin and User
'''
@app.route("/")
@app.route("/index")
def home():
    return render_home(isAdmin=False)


@app.route("/admin")
def admin_home():
    return render_home(isAdmin=True)


'''
Landing page after session Logout
'''
@app.route("/logout", methods=["GET", "POST"])
def logout():
    global currFlag
    currFlag = False

    sessionId = request.args.get("sessionId")

    if sessionId:
        insertLogs(sessionId)
    global session
    session = {}
    return render_template("index.html")


'''
Session List for Student
'''
@app.route("/sessionList.html", methods=["POST"])
def sessionList():
    if "user" in session:
        uname = session["user"].username
        passwd = session["user"].password
        changePass = False
    else:
        uname = request.form.get("uname")
        passwd = request.form.get("pass")
        changePass = True if request.form.get(
            "changePass", False) == "on" else False

    user = User.query.filter_by(
        username=uname, password=passwd, isAdmin=False).first()

    if user:
        if changePass:
            user.password = request.form.get("newPass", False)
            db.session.commit()
        data = getSessionData()
        session["user"] = user
        return render_template("sessionList.html", data={"user": session["user"], "sessions": data})
    flash("Login Unsuccessful. Please check username and password", "danger")
    return redirect(url_for("home"))


'''
Session List for Admin
'''
@app.route("/sessionListAdmin.html", methods=["GET", "POST"])
def sessionListAdmin():
    if "user" in session:
        uname = session["user"].username
        passwd = session["user"].password
        changePass = False
    else:
        uname = request.form.get("uname")
        passwd = request.form.get("pass")
        changePass = True if request.form.get(
            "changePass", False) == "on" else False

    user = User.query.filter_by(
        username=uname, password=passwd, isAdmin=True).first()

    if user:
        if changePass:
            user.password = request.form.get("newPass", False)
            db.session.commit()
        session["user"] = user
        data = getSessionData()
        return render_template(
            "sessionListAdmin.html",
            data={"user": session["user"], "sessions": data},
        )
    flash("Login Unsuccessful. Please check username and password", "danger")
    return redirect(url_for("admin_home"))


'''
Add New User/Admin
'''
@app.route("/addUser.html")
def addUser():
    return render_template("addUser.html", data={"user": session["user"]})


'''
Add User
'''
@app.route("/handle_data_add_user", methods=["POST"])
def handle_data_add_user():
    fname = request.form.get("fname", "")
    lname = request.form.get("lname", "")
    userId = request.form.get("userId", "")
    userEmail = request.form.get("userEmail", "")
    isAdmin = True if request.form.get("isAdmin", False) == "on" else False
    try:
        user = User(username=f"{userId}", email=userEmail, password="asd",
                    name=f"{fname} {lname}", isAdmin=isAdmin, isFlagged=False)
        db.session.add(user)
        db.session.commit()
    except Exception as e:
        print(e)
        # send error message on UI
    return studentList()


'''
Add new Exam Session
'''
@app.route("/createSession.html")
def createSession():
    return render_template("createSession.html", data={"user": session["user"]})


'''
Add Exam Session
'''
@app.route("/handle_data_create_session", methods=["POST"])
def handle_data_create_session():
    strtDate = datetime.strptime(
        request.form.get("strtDate", ""), "%Y-%m-%dT%H:%M")
    endDate = datetime.strptime(
        request.form.get("endDate", ""), "%Y-%m-%dT%H:%M")
    sessName = request.form.get("sessionName", "")
    sessionNotes = request.form.get("sessionNotes", "")
    try:
        session = Session(sessionname=sessName, startdate=strtDate,
                          enddate=endDate, sessionnotes=sessionNotes)
        db.session.add(session)
        db.session.commit()
    except Exception as e:
        print(e)

    return sessionListAdmin()


'''
Display Session Details
'''
@app.route("/showSessionData.html")
def showSessionData():
    sessionId = request.args.get("sessionId")
    currsession = Session.query.filter_by(id=sessionId).first()
    session["currSession"] = currsession
    return render_template("showSessionData.html", data={"user": session["user"], "sessionData": currsession})


'''
Delete Session
'''
@app.route("/deleteSession.html", methods=["POST"])
def deleteSession():
    Session.query.filter_by(id=session["currSession"].id).delete()
    db.session.commit()
    del session["currSession"]
    return redirect(url_for("sessionListAdmin"))


'''
Update/Delete User information
'''
@app.route("/updateUser.html", methods=["POST"])
def updateUser():
    if request.form["action"] == "toggleFlag":
        uid = request.form.get("userId")
        user = User.query.filter_by(id=session["currstdId"].id).first()
        user.isFlagged = not user.isFlagged
        db.session.commit()
    elif request.form["action"] == "delUser":
        User.query.filter_by(id=session["currstdId"].id).delete()
        db.session.commit()
    del session["currstdId"]
    return redirect(url_for("studentList"))


'''
Show User Details
'''
@app.route("/showUserData.html", methods=["GET", "POST"])
def showUserData():
    stdId = request.args.get("studentId")
    user = User.query.filter_by(id=stdId).first()
    logs = Logs.query.filter_by(userId=stdId).all()
    session["currstdId"] = user
    return render_template("showUserData.html", data={"user": session["user"], "student": user, "logs": logs})


'''
Display Student List to Admin
'''
@app.route("/studentList.html")
def studentList():

    data = User.query.filter_by(isAdmin=False).all()
    # TODO check if data is not empty
    return render_template(
        "studentList.html",
        data={"user": session["user"], "students": data},
    )


'''
Start Exam Session
'''
@app.route("/examPage.html", methods=["POST", "GET"])
def examPage():
    sessionId = request.args.get("sessionId")
    return render_template("examPage.html", data={"user": session["user"], "sessionId": sessionId})


'''
Invoke ML models
'''
@app.route("/run_model")
def run_model():
    video = cv2.VideoCapture(0)
    return Response(invoke_models(video), mimetype="multipart/x-mixed-replace; boundary=frame")


def invoke_models(video):
    try:
        global currFlag
        currFlag = True
        eye_tracker = EyeTracker(video)
        mouth_open = Mouth_Opening(video)
        head_pos = Head_Position(video)
        object_detector = Object_Detector(video)

        logger = {"head_logger": [], "mouth_logger": [],
                  "phone_logger": [], "eye_logger": []}
        # timer()

        while currFlag:
            logger["head_logger"].extend(head_pos.head_position())
            logger["mouth_logger"].extend(mouth_open.mouth_opening())
            logger["eye_logger"].extend(eye_tracker.eye_detector())
            logger["phone_logger"].extend(object_detector.person_and_phone())
            logger["userInfo"] = {}
            logger["userInfo"]["id"] = session["user"].id
            logger["userInfo"]["username"] = session["user"].username
            checkForFlag(logger)
            # json_object=json.dumps(logger, indent = 4)
            with open("log.json", "w") as outfile:
                json.dump(logger, outfile)
    except Exception as e:
        print(e)


# HELPER FUNCTIONS
'''
Insert logs in Data Base
'''


def insertLogs(sessionId):
    try:
        data = None
        with open("log.json") as f:
            data = json.loads(f.read())
    except Exception as e:
        print(e)
    if data:
        try:
            sessiondata = Session.query.filter_by(id=sessionId).first()
            logs = Logs(
                rawData=str.encode(json.dumps(data)),
                userId=session["user"].id,
                sessionid=sessiondata.id,
                sessionname=sessiondata.sessionname,
                timestamp=datetime.now(),
            )
            db.session.add(logs)
            db.session.commit()
            f = open("log.json", "w")
            f.truncate(0)
        except Exception as e:
            print(e)


'''
Flag student
'''


def checkForFlag(logs):
    if len(logs["head_logger"]) > 100 or len(logs["mouth_logger"]) > 100 or len(logs["eye_logger"]) > 100 or len(logs["phone_logger"]) > 10:
        user = User.query.filter_by(username=session["user"].username).first()
        user.isFlagged = True
        db.session.commit()


'''
Render HomePage
'''


def render_home(isAdmin=False):
    global currFlag
    currFlag = False
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


def getSessionData():
    return Session.query.all()
