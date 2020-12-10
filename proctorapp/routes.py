import cv2
import json
import threading
from flask import Flask, Response, render_template, url_for
from proctorapp import app
from proctorapp.forms import RegistrationForm, LoginForm
from proctorapp.proctoring_models import BaseModel, EyeTracker, Mouth_Opening, Head_Position, Object_Detector
from flask import request


@app.route("/")
@app.route("/index")
def home():
    return render_template("index.html")


@app.route("/admin")
def admin_home():
    form = LoginForm()
    return render_template("indexAdmin.html", form=form)


@app.route("/sessionList.html")
def sessionList():

    return render_template(
        "sessionList.html",
        data={
            "user": {"name": "Pranay", "id": "prverma", "isAdmin": False},
            "sessions": [
                {"sessionName": "OOD Exam", "date": "November 16th 6:00 pm - 8:00 pm"},
                {"sessionName": "ANN Exam", "date": "November 17th 6:00 pm - 8:00 pm"},
                {"sessionName": "NLP Exam", "date": "November 18th 6:00 pm - 8:00 pm"},
            ],
        },
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
    return render_template(
        "sessionListAdmin.html",
        data={
            "user": {"name": "Pranay", "id": "prverma", "isAdmin": True},
            "sessions": [
                {"sessionName": "OOD Exam", "date": "November 16th 6:00 pm - 8:00 pm"},
                {"sessionName": "ANN Exam", "date": "November 17th 6:00 pm - 8:00 pm"},
                {"sessionName": "NLP Exam", "date": "November 18th 6:00 pm - 8:00 pm"},
            ],
        },
    )


@app.route("/studentList.html")
def studentList():
    return render_template(
        "studentList.html",
        data={"user": {"name": "Pranay", "id": "prverma", "isAdmin": True}, "students": [{"name": "bahadur"}, {"name": "loda"}, {"name": "chutiya"}]},
    )


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
    return studentList()


@app.route("/handle_data_create_session", methods=["GET", "POST"])
def handle_data_create_session():
    strtDate = request.form.get("strtDate", "")
    endDate = request.form.get("endDate", "")
    sessionName = request.form.get("sessionName", "")
    sessionNotes = request.form.get("sessionNotes", "")
    return sessionListAdmin()


def invoke_models(video):
    try:
        eye_tracker = EyeTracker(video)
        mouth_open = Mouth_Opening(video)
        head_pos = Head_Position(video)
        object_detector = Object_Detector(video)

        logger = {"head_logger": [], "mouth_logger": [], "phone_logger": [], "eye_logger": []}
        timer()
        while True:
            logger["head_logger"].extend(head_pos.head_position())
            logger["mouth_logger"].extend(mouth_open.mouth_opening())
            logger["eye_logger"].extend(eye_tracker.eye_detector())
            logger["phone_logger"].extend(object_detector.person_and_phone())
            # json_object=json.dumps(logger, indent = 4)
            with open("log.json", "w") as outfile:
                json.dump(logger, outfile)
    except Exception as e:
        print(e)


def hello():
    print("hello, world")
    timer()


def timer():
    threading.Timer(10.0, hello).start()