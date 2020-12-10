import cv2
import json
from flask import Flask, Response, render_template, url_for
from proctorapp import app
from proctorapp.forms import RegistrationForm, LoginForm
from proctorapp.proctoring_models import BaseModel, EyeTracker, Mouth_Opening, Head_Position, Object_Detector


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
    return render_template("sessionListAdmin.html", data={"user": {"name": "Pranay", "id": "prverma", "isAdmin": True}})


@app.route("/studentList.html")
def studentList():
    return render_template("studentList.html", data={"user": {"name": "Pranay", "id": "prverma", "isAdmin": True}})


@app.route("/examPage.html")
def examPage():
    return render_template('examPage.html')

@app.route('/run_model')
def run_model():
    try:
        video = cv2.VideoCapture(0)
        eye_tracker = EyeTracker(video)
        mouth_open = Mouth_Opening(video)
        head_pos = Head_Position(video)
        object_detector = Object_Detector(video)

        logger = {'head_logger': [], 'mouth_logger': [], 'phone_logger': [], 'eye_logger': []}

        while True:
            logger['head_logger'].extend(head_pos.head_position())
            logger['mouth_logger'].extend(mouth_open.mouth_opening())
            logger['phone_logger'].extend(object_detector.person_and_phone())
            logger['eye_logger'].extend(eye_tracker.eye_detector())
            # json_object=json.dumps(logger, indent = 4)
            with open("log.json", "w") as outfile:
                json.dump(logger, outfile)

    except Exception as e:
        print(e)

    # video = cv2.VideoCapture(0)
    # return Response(base_model(video),
    #                 mimetype='multipart/x-mixed-replace; boundary=frame')
