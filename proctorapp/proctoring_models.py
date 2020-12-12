import cv2
import math
import os.path
from os import path
import numpy as np
from datetime import datetime
from proctorapp import yolov3
import wget
from proctorapp.mlmodel.face_detector import get_face_detector, find_faces
from proctorapp.mlmodel.face_landmarks import get_landmark_model, detect_marks, draw_marks


def getCurrentTime():
    return datetime.now().strftime("%m/%d/%Y, %H:%M:%S")


class BaseModel:
    def __init__(self, video):
        self.video = video
        self.face_model = get_face_detector()
        self.landmark_model = get_landmark_model()
        self.ret, self.img = self.video.read()
        self.size = self.img.shape
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                # Left eye left corner
                (-225.0, 170.0, -135.0),
                # Right eye right corne
                (225.0, 170.0, -135.0),
                # Left Mouth corner
                (-150.0, -150.0, -125.0),
                # Right mouth corner
                (150.0, -150.0, -125.0),
            ]
        )
        # Camera internals
        self.focal_length = self.size[1]
        self.center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array([[self.focal_length, 0, self.center[0]], [0, self.focal_length, self.center[1]], [0, 0, 1]], dtype="double")

        # mouth
        self.outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
        self.d_outer = [0] * 5
        self.inner_points = [[61, 67], [62, 66], [63, 65]]
        self.d_inner = [0] * 3
        # eye tracker
        self.left = [36, 37, 38, 39, 40, 41]
        self.right = [42, 43, 44, 45, 46, 47]
        self.kernel = np.ones((9, 9), np.uint8)
        self.logger = {"head_logger": [], "mouth_logger": [], "phone_logger": [], "eye_logger": []}


class EyeTracker(BaseModel):
    def __init__(self, video):
        super().__init__(video)

    def eye_detector(self):
        ret, img = self.video.read()
        rects = find_faces(img, self.face_model)
        eye_logger = []
        for rect in rects:
            shape = detect_marks(img, self.landmark_model, rect)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask, end_points_left = self.eye_on_mask(mask, self.left, shape)
            mask, end_points_right = self.eye_on_mask(mask, self.right, shape)
            mask = cv2.dilate(mask, self.kernel, 5)

            eyes = cv2.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            mid = int((shape[42][0] + shape[39][0]) // 2)
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(eyes_gray, 50, 255, cv2.THRESH_BINARY)
            thresh = self.process_thresh(thresh)

            eyeball_pos_left = self.contouring(thresh[:, 0:mid], mid, img, end_points_left)
            eyeball_pos_right = self.contouring(thresh[:, mid:], mid, img, end_points_right, True)
            eye_logger.extend(self.print_eye_pos(img, eyeball_pos_left, eyeball_pos_right))
        return eye_logger

    def eye_on_mask(self, mask, side, shape):
        """
        Create ROI on mask of the size of eyes and also find the extreme points of each eye

        Parameters
        ----------
        mask : np.uint8
            Blank mask to draw eyes on
        side : list of int
            the facial landmark numbers of eyes
        shape : Array of uint32
            Facial landmarks

        Returns
        -------
        mask : np.uint8
            Mask with region of interest drawn
        [l, t, r, b] : list
            left, top, right, and bottommost points of ROI

        """
        points = [shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)
        l = points[0][0]
        t = (points[1][1] + points[2][1]) // 2
        r = points[3][0]
        b = (points[4][1] + points[5][1]) // 2
        return mask, [l, t, r, b]

    def contouring(self, thresh, mid, img, end_points, right=False):

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            cnt = max(cnts, key=cv2.contourArea)
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if right:
                cx += mid
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
            pos = find_eyeball_position(end_points, cx, cy)
            return pos
        except:
            pass

    def process_thresh(self, thresh):
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        thresh = cv2.medianBlur(thresh, 3)
        thresh = cv2.bitwise_not(thresh)
        return thresh

    def print_eye_pos(self, img, left, right):
        eye_logger = []
        if left == right and left != 0:
            text = ""
            if left == 1:
                eye_logger.append({"exception": "looking left", "timestamp": getCurrentTime()})
            elif left == 2:
                eye_logger.append({"exception": "looking right", "timestamp": getCurrentTime()})
            elif left == 3:
                eye_logger.append({"exception": "looking up", "timestamp": getCurrentTime()})
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, text, (30, 30), font,1, (0, 255, 255), 2, cv2.LINE_AA)
        return eye_logger

    def find_eyeball_position(self, end_points, cx, cy):
        """Find and return the eyeball positions, i.e. left or right or top or normal"""
        x_ratio = (end_points[0] - cx) / (cx - end_points[2])
        y_ratio = (cy - end_points[1]) / (end_points[3] - cy)
        if x_ratio > 3:
            return 1
        elif x_ratio < 0.33:
            return 2
        elif y_ratio < 0.33:
            return 3
        else:
            return 0


class Mouth_Opening(BaseModel):
    def __init__(self, video):
        super().__init__(video)

    def mouth_opening(self):
        mouth_logger = []
        ret, img = self.video.read()
        rects = find_faces(img, self.face_model)
        for rect in rects:
            shape = detect_marks(img, self.landmark_model, rect)
            cnt_outer = 0
            cnt_inner = 0
            draw_marks(img, shape[48:])
            for i, (p1, p2) in enumerate(self.outer_points):
                if self.d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                    cnt_outer += 1
            for i, (p1, p2) in enumerate(self.inner_points):
                if self.d_inner[i] + 2 < shape[p2][1] - shape[p1][1]:
                    cnt_inner += 1
            if cnt_outer > 3 and cnt_inner > 2:
                mouth_logger.append({"exception": "Mouth Open", "timestamp": getCurrentTime()})
                # print('Mouth open')
        return mouth_logger


class Head_Position(BaseModel):
    def __init__(self, video):
        super().__init__(video)

    def head_position(self):
        head_pos_logger = []
        ret, img = self.video.read()
        # if ret == True:
        faces = find_faces(img, self.face_model)
        for face in faces:
            marks = detect_marks(img, self.landmark_model, face)
            # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
            image_points = np.array(
                [
                    marks[30],  # Nose tip
                    marks[8],  # Chin
                    marks[36],  # Left eye left corner
                    marks[45],  # Right eye right corne
                    marks[48],  # Left Mouth corner
                    marks[54],  # Right mouth corner
                ],
                dtype="double",
            )
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points, image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP
            )

            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose

            (nose_end_point2D, jacobian) = cv2.projectPoints(
                np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, self.camera_matrix, dist_coeffs
            )

            # for p in image_points:
            #     cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = self.head_pose_points(img, rotation_vector, translation_vector, self.camera_matrix)

            try:
                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90

            try:
                m = (x2[1] - x1[1]) / (x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1 / m)))
            except:
                ang2 = 90

            if ang1 >= 48:
                head_pos_logger.append({"exception": "Head Down", "timestamp": getCurrentTime()})
            elif ang1 <= -48:
                head_pos_logger.append({"exception": "Head Up", "timestamp": getCurrentTime()})

            if ang2 >= 48:
                head_pos_logger.append({"exception": "Head Right", "timestamp": getCurrentTime()})
            elif ang2 <= -48:
                head_pos_logger.append({"exception": "Head Left", "timestamp": getCurrentTime()})
        return head_pos_logger

    def head_pose_points(self, img, rotation_vector, translation_vector, camera_matrix):
        """
        Get the points to estimate head pose sideways

        Parameters
        ----------
        img : np.unit8
            Original Image.
        rotation_vector : Array of float64
            Rotation Vector obtained from cv2.solvePnP
        translation_vector : Array of float64
            Translation Vector obtained from cv2.solvePnP
        camera_matrix : Array of float64
            The camera matrix

        Returns
        -------
        (x, y) : tuple
            Coordinates of line to estimate head pose

        """
        rear_size = 1
        rear_depth = 0
        front_size = img.shape[1]
        front_depth = front_size * 2
        val = [rear_size, rear_depth, front_size, front_depth]
        point_2d = self.get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
        y = (point_2d[5] + point_2d[8]) // 2
        x = point_2d[2]

        return (x, y)

    def get_2d_points(self, img, rotation_vector, translation_vector, camera_matrix, val):
        """Return the 3D points present as 2D for making annotation box"""
        point_3d = []
        dist_coeffs = np.zeros((4, 1))
        rear_size = val[0]
        rear_depth = val[1]
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = val[2]
        front_depth = val[3]
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d img points
        (point_2d, _) = cv2.projectPoints(point_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        point_2d = np.int32(point_2d.reshape(-1, 2))
        return point_2d


class Object_Detector(BaseModel):
    def __init__(self, video):
        if not path.exists("./proctorapp/mlmodel/models/yolov3.weights"):
            _ = wget.download("https://pjreddie.com/media/files/yolov3.weights", out="./proctorapp/mlmodel/models/yolov3.weights")
        super().__init__(video)
        self.yolo = yolov3.YoloV3()
        yolov3.load_darknet_weights(self.yolo, "./proctorapp/mlmodel/models/yolov3.weights")

    def person_and_phone(self):
        phone_logger = []
        ret, image = self.video.read()
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 320))
        img = img.astype(np.float32)
        img = np.expand_dims(img, 0)
        img = img / 255
        class_names = [c.strip() for c in open("./proctorapp/mlmodel/models/classes.TXT").readlines()]
        try:
            boxes, scores, classes, nums = self.yolo(img)
        except Exception as e:
            print(e)

        count = 0
        for i in range(nums[0]):
            if int(classes[0][i] == 0):
                count += 1
            if int(classes[0][i] == 67):
                phone_logger.append({"exception": "Mobile Phone detected", "timestamp": getCurrentTime()})
        if count == 0:
            phone_logger.append({"exception": "No person detected", "timestamp": getCurrentTime()})
        elif count > 1:
            phone_logger.append({"exception": "More than one person detected", "timestamp": getCurrentTime()})
        return phone_logger
