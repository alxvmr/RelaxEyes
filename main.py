### Библиотека RelaxEye

import mediapipe as mp
import cv2
import dlib
from imutils import face_utils
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import opencv_plot
from opencv_plot import Plotter
import random
import math
from scipy.spatial import distance as dist
import keyboard
import sys

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5)

path_face_dots = 'shape_predictor_68_face_landmarks.dat'
face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(path_face_dots)

POSE_LANDMARK_1 = 11
POSE_LANDMARK_2 = 12

EYE_LANDMARK_1 = 36
EYE_LANDMARK_2 = 45

def write_list_to_file (lines, file):
    with open(file, "w") as f:
        for line in lines:
            f.write(str(line) + '\n')

def write_elem_to_file (elem, file): #дозаписываем элемент в файл
    with open(file, "a") as f:
        f.write(str(elem) + '\n')
    f.close()

def get_list_from_file(file):
    el = []
    with open(file) as f:
        for line in f:
            el.append(float(line))
    f.close()
    return el

def eye_aspect_ratio(eye):
    # Рассчитать евклидово расстояние между двумя наборами
    # Вертикальные координаты метки глаза (X, Y)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Рассчитать евклидово расстояние между уровнями
    # Горизонтальные координаты метки глаза (X, Y)
    C = dist.euclidean(eye[0], eye[3])
    # Расчет соотношения глаз
    ear = (A + B) / (2.0 * C)
    # Вернуться к соотношению сторон очков
    return ear


def screen_proximity_tracking(show_RTgr = False, show_video = False, file_output='distance_between_eye_points.txt'):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    length_line_eye = []  # расстояние между точками глаз
    open(file_output, 'w').close()  # очистили файл перед записью

    if (show_RTgr):
        p = Plotter(400, 1000)

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # сделаем детектор
        result = mp_pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Точки плеч
        landmark_shl = [
            result.pose_landmarks.landmark[POSE_LANDMARK_1],
            result.pose_landmarks.landmark[POSE_LANDMARK_2]
        ]

        # денормализуем координаты
        sh_list = []  # список, хранящий денормализованные координаты плеч
        for landmark in landmark_shl:
            x = landmark.x
            y = landmark.y

            shape_sh = image.shape
            relative_x = int(x * shape_sh[1])
            relative_y = int(y * shape_sh[0])

            if (show_video):
                sh_list.append([relative_x, relative_y])
                cv2.circle(image, (relative_x, relative_y), 5, (0, 0, 255), -1)
        if (show_video):
            cv2.line(image, tuple(sh_list[0]), tuple(sh_list[1]), (0, 0, 255), 1)  # соединим точки плеч (2 линия)

        ################## ЛИЦО ################################
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detector(gray, 0)

        # нужные точки лица
        for (i, rect) in enumerate(rects):
            shape = face_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)  # массив точек лица
            for i in [36, 45]:  # отрисовываем 2 нужные точки
                x, y = shape[i]
                if (show_video):
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            length_line = cv2.norm(shape[EYE_LANDMARK_1], shape[EYE_LANDMARK_2])
            if (show_video):
                cv2.line(image,
                         tuple(shape[EYE_LANDMARK_1]),
                         tuple(shape[EYE_LANDMARK_2]),
                         (0, 255, 0),
                         1)  # соединим точки глаз (1 линия)
                # вывод длины линии над линией и добавление длины в массив
                cv2.putText(
                    image,
                    str(length_line),  # длина линии (L-2 норма)
                    ( round((shape[EYE_LANDMARK_1][0] + shape[EYE_LANDMARK_2][0]) / 2),
                      round((shape[EYE_LANDMARK_1][1] + shape[EYE_LANDMARK_2][1]) / 2) ),
                    # position at which writing has to start
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (209, 80, 0, 255),
                    1
                )

            if (show_RTgr):
                p.plot(length_line, "Distance between eye points")  # отрисовка точки на графике

            write_elem_to_file(length_line, file_output)
            length_line_eye.append(length_line)

        if (show_video):
            cv2.imshow('Video', image)
        k = cv2.waitKey(10)

        if keyboard.is_pressed('Ctrl + Q'):  # Остановить цикл
            cap.release()
            cv2.destroyAllWindows()
            break

    return length_line_eye


def tracking_eye_shoulders(show_RTgr = False, show_video = False, file_output='distance_between_eye_shoulders.txt'):
    cap = cv2.VideoCapture(0)
    length_line_between_shoulders_ear = []  # расстояние между точками плеч
    open(file_output, 'w').close()  # очистили файл перед записью
    global shape

    if (show_RTgr):
        p = Plotter(400, 1000)

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ################## ЛИЦО ################################
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detector(gray, 0)

        # нужные точки лица
        for (i, rect) in enumerate(rects):
            shape = face_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)  # массив точек лица
            for i in [EYE_LANDMARK_1, EYE_LANDMARK_2]:  # отрисовываем 2 нужные точки
                x, y = shape[i]
                if (show_video):
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            if (show_video):
                cv2.line(image,
                         tuple(shape[EYE_LANDMARK_1]),
                         tuple(shape[EYE_LANDMARK_2]),
                         (0, 255, 0),
                         1)  # соединим точки глаз (1 линия)

        ################## ПЛЕЧИ ################################

        # сделаем детектор
        result = mp_pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Точки плеч
        landmark_shl = [
            result.pose_landmarks.landmark[POSE_LANDMARK_1],
            result.pose_landmarks.landmark[POSE_LANDMARK_2]
        ]

        # денормализуем координаты
        sh_list = []  # список, хранящий денормализованные координаты плеч
        for landmark in landmark_shl:
            x = landmark.x
            y = landmark.y

            shape_sh = image.shape

            relative_x = int(x * shape_sh[1])
            relative_y = int(y * shape_sh[0])

            sh_list.append([relative_x, relative_y])
            if (show_video):
                cv2.circle(image, (relative_x, relative_y), 5, (0, 0, 255), -1)
        if (show_video):
            cv2.line(image, tuple(sh_list[0]), tuple(sh_list[1]), (0, 0, 255), 1)  # соединим точки плеч (2 линия)

        # серединные точки shape[0], shape[2]
        eye_midpoints = [ round((shape[EYE_LANDMARK_1][0] + shape[EYE_LANDMARK_2][0]) / 2)
                        , round((shape[EYE_LANDMARK_1][1] + shape[EYE_LANDMARK_2][1]) / 2) ]
        shoulder_midpoints = [ round((sh_list[0][0] + sh_list[1][0]) / 2)
                             , round((sh_list[0][1] + sh_list[1][1]) / 2) ]

        # построим линию по серединным точкам
        if (show_video):
            cv2.circle(image, tuple(eye_midpoints), 5, (255, 0, 0), -1)
            cv2.circle(image, tuple(shoulder_midpoints), 5, (255, 0, 0), -1)
            cv2.line(image, tuple(eye_midpoints), tuple(shoulder_midpoints), (255, 0, 0), 1)

        # вывод длины линии над линией и добавление длины в массив
        length_line = cv2.norm(np.array(eye_midpoints), np.array(shoulder_midpoints))
        length_line_between_shoulders_ear.append(length_line)
        if (show_video):
            cv2.putText(
                image,
                str(length_line),  # Длина линии (L-2 норма)
                (round((eye_midpoints[0] + shoulder_midpoints[0]) / 2),
                 round((eye_midpoints[1] + shoulder_midpoints[1]) / 2)),  # position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (209, 80, 0, 255),
                1
            )

        write_elem_to_file(length_line, file_output)

        if (show_RTgr):
            p.plot(length_line, "Distance between shoulders")  # отрисовка точки на графике

        if (show_video):
            cv2.imshow('Raw Webcam Feed', image)

        k = cv2.waitKey(10)
        if keyboard.is_pressed('Ctrl + Q'):  # Остановить цикл
            cap.release()
            cv2.destroyAllWindows()
            break

    return length_line_between_shoulders_ear


def tracking_blink_squint(EYE_AR_THRESH = 0.25, BLINK_TRESH = 0.19, DURATION = 0, WAIT_TIME = 2, show_video = False, file_output='ear.txt'):
    TOTAL_COUNT_BLINKING = 0
    IS_BLINKING = False
    ear_arr = []
    open(file_output, 'w').close()  # очистили файл перед записью

    # получаем точки глаз
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ################## ЛИЦО ################################
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detector(gray, 0)

        # нужные точки лица
        for (i, rect) in enumerate(rects):
            shape = face_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)  # массив точек лица
            for i in range(36, 48):  # отрисовываем нужные точки глаз
                x, y = shape[i]
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # Среднее соотношение сторон обоих глаз
            ear = (leftEAR + rightEAR) / 2.0

            ear_arr.append(ear)
            write_elem_to_file(ear, file_output)

            if(show_video):
                cv2.putText(image,
                            "EAR: {:.2f}".format(ear),
                            (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2)
            if (ear <= EYE_AR_THRESH):
                DURATION += 1
                if (DURATION > WAIT_TIME):
                    if (show_video):
                        cv2.putText(image, "Don't squint!", (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif ((IS_BLINKING == False) & (ear <= BLINK_TRESH)):
                    TOTAL_COUNT_BLINKING += 1
                    IS_BLINKING = True
            else:
                DURATION = 0
                IS_BLINKING = False

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if (show_video):
            cv2.putText(image,
                        "Blinks: {}".format(TOTAL_COUNT_BLINKING),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2)
            cv2.imshow('Raw Webcam Feed', image)

        k = cv2.waitKey(10)
        if keyboard.is_pressed('Ctrl + Q'):  # Остановить цикл
            # последняя строка файла - количество морганий
            write_elem_to_file(TOTAL_COUNT_BLINKING, file_output)
            cap.release()
            cv2.destroyAllWindows()
            break

    return ear_arr


#Тестовый запуск

if (__name__ == '__main__'):
    #arr = screen_proximity_tracking(show_RTgr=False, show_video=False)
    #arr = tracking_eye_shoulders(show_RTgr=False, show_video=False)
    #arr = tracking_blink_squint(show_video=True, BLINK_TRESH = 0.25)

    # plt.title('Изменение расстояния между точками глаз')
    # plt.xlabel('Время')
    # plt.ylabel('Расстояние между точками глаз')
    # plt.plot(arr)
    # plt.show()

    #запуск нескольких функций одновременно
    #ret_id1 = screen_proximity_tracking(show_RTgr=False, show_video=False).remote()