### Библиотека RelaxEye
import mediapipe as mp
import cv2
import dlib
from imutils import face_utils
import numpy as np
from opencv_plot import Plotter
from scipy.spatial import distance as dist
import keyboard
import matplotlib.pyplot as plt

'''
Настройка моделей
'''

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

duration = 0  # количество итераций цикла с выполненными условиями
TOTAL_COUNT_BLINKING = 0  # количество морганий
IS_BLINKING = False  # есть ли моргание в текущий момент

'''
Вход: массив, путь к файлу
Запись в файл массива <lines> в файл <file>
'''


def write_list_to_file(lines, file):
    with open(file, "w") as f:
        for line in lines:
            f.write(str(line) + '\n')


'''
Вход: элемент, путь к файлу
Запись элемента <elem> в файл <file>
'''


def write_elem_to_file(elem, file):  # дозаписываем элемент в файл
    with open(file, "a") as f:
        f.write(str(elem) + '\n')
    f.close()


'''
Вход: путь к файлу
Получение массива <el> из файла <file>
Возврат: полученный массив
'''


def get_list_from_file(file):
    el = []
    with open(file) as f:
        for line in f:
            if line != "None\n":
                el.append(float(line))
    f.close()
    return el


'''
Вход: массив точек глаза
Возврат: соотношение сторон глаза
'''


def eye_aspect_ratio(eye):
    # Рассчитать евклидово расстояние между двумя наборами
    # Вертикальные координаты метки глаза (X, Y)
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    # Рассчитать евклидово расстояние между уровнями
    # Горизонтальные координаты метки глаза (X, Y)
    c = dist.euclidean(eye[0], eye[3])
    # Расчет соотношения глаз
    ear = (a + b) / (2.0 * c)
    return ear


'''
РАСЧЕТ РАССТОЯНИЯ МЕЖДУ ГЛАЗАМИ
Вход:
    cap - текущий кадр
    p - объект класса Plotter (для вывода real-time графика)
    show_rtgr - true - выводим real-time график, false - не выводим
    show_video - true - выводим видео с точками и линиями, false - не выводим
Возврат: расстояние между глазами в текущий момент времени / None
'''


def screen_proximity_tracking(cap, p, show_rtgr=False, show_video=False):
    length_line = None
    ret, frame = cap.read()
    image = frame

    '''
    Блок работы с точками лица
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 0)

    # нужные точки лица
    for (i, rect) in enumerate(rects):
        shape = face_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)  # массив точек лица

        if show_video:
            for j in [36, 45]:  # отрисовываем 2 нужные точки
                x, y = shape[j]
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        length_line = cv2.norm(shape[EYE_LANDMARK_1], shape[EYE_LANDMARK_2])  # расстояние между точками глаз
        if show_video:
            cv2.line(image,
                     tuple(shape[EYE_LANDMARK_1]),
                     tuple(shape[EYE_LANDMARK_2]),
                     (0, 255, 0),
                     1)  # соединим точки глаз (1 линия)
            # вывод длины линии над линией и добавление длины в массив
            cv2.putText(
                image,
                str(length_line),  # длина линии (L-2 норма)
                (round((shape[EYE_LANDMARK_1][0] + shape[EYE_LANDMARK_2][0]) / 2),
                 round((shape[EYE_LANDMARK_1][1] + shape[EYE_LANDMARK_2][1]) / 2)),
                # position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (209, 80, 0, 255),
                1
            )

        if show_rtgr:
            p.plot(length_line, "Distance between eye points")  # отрисовка точки на графике

        # write_elem_to_file(length_line, file_output)

    if show_video:
        cv2.imshow('Video', image)
    cv2.waitKey(10)

    return length_line


'''
РАСЧЕТ РАССТОЯНИЕ МЕЖДУ ЛИНИЕЙ ГЛАЗ И ЛИНИЕЙ ПЛЕЧ
Вход:
    cap - текущий кадр
    p - объект класса Plotter (для вывода real-time графика)
    show_rtgr - true - выводим real-time график, false - не выводим
    show_video - true - выводим видео с точками и линиями, false - не выводим
Возврат: расстояниме между линией глаз и линией плеч в текущий момент времени / None
'''


def tracking_eye_shoulders(cap, p, show_rtgr=False, show_video=False):
    global shape
    length_line = None
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    '''
    Блок работы с точками лица
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 0)

    if len(rects) == 0:
        return None

    # нужные точки лица
    for (i, rect) in enumerate(rects):
        shape = face_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)  # массив точек лица

        if show_video:
            for j in [EYE_LANDMARK_1, EYE_LANDMARK_2]:  # отрисовываем 2 нужные точки
                x, y = shape[j]
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.line(image,
                     tuple(shape[EYE_LANDMARK_1]),
                     tuple(shape[EYE_LANDMARK_2]),
                     (0, 255, 0),
                     1)  # соединим точки глаз (1 линия)

    '''
    Блок работы с точками плеч
    '''
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
        if show_video:
            cv2.circle(image, (relative_x, relative_y), 5, (0, 0, 255), -1)
    if show_video:
        cv2.line(image, tuple(sh_list[0]), tuple(sh_list[1]), (0, 0, 255), 1)  # соединим точки плеч (2 линия)

    # серединные точки shape[0], shape[2]
    eye_midpoints = [round((shape[EYE_LANDMARK_1][0] + shape[EYE_LANDMARK_2][0]) / 2),
                     round((shape[EYE_LANDMARK_1][1] + shape[EYE_LANDMARK_2][1]) / 2)]
    shoulder_midpoints = [round((sh_list[0][0] + sh_list[1][0]) / 2),
                          round((sh_list[0][1] + sh_list[1][1]) / 2)]

    # построим линию по серединным точкам
    if show_video:
        cv2.circle(image, tuple(eye_midpoints), 5, (255, 0, 0), -1)
        cv2.circle(image, tuple(shoulder_midpoints), 5, (255, 0, 0), -1)
        cv2.line(image, tuple(eye_midpoints), tuple(shoulder_midpoints), (255, 0, 0), 1)

    # вывод длины линии над линией и добавление длины в массив
    length_line = cv2.norm(np.array(eye_midpoints), np.array(shoulder_midpoints))
    if show_video:
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

    # write_elem_to_file(length_line, file_output)

    if show_rtgr:
        p.plot(length_line, "Distance between shoulders")  # отрисовка точки на графике

    if show_video:
        cv2.imshow('Raw Webcam Feed', image)
    cv2.waitKey(10)

    return length_line


'''
РАСЧЕТ СООТНОШЕНИЯ СТОРОН ГЛАЗ, ОПРЕДЕЛЕНИЕ КОЛИЧЕСТВА МОРГАНИЙ И ПРИЩУРА
Вход:
    cap - текущий кадр
    EYE_AR_THRESH - пороговое значение, при котором выявляется прищур
    BLINK_TRESH - пороговое значение, при котором выявляется моргание
    WAIT_TIME - количество итераций цикла, при которых значение должно удовлетворять условию морагия/прищура
    show_video - true - выводим обработанное видео, false - не выводим
    file_output - файл, куда будет записываться соотношение сторон глаз (по умолчанию ear.txt)
Возврат:
    соотношение сторон глаз в настоящий момент / None
'''


def tracking_blink_squint(cap, EYE_AR_THRESH=0.25, BLINK_THRESH=0.19, WAIT_TIME=2, show_video=False):
    global duration, TOTAL_COUNT_BLINKING, IS_BLINKING
    ear = None
    # получаем точки глаз
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    '''
    Блок работы с точками лица
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 0)

    # нужные точки лица
    for (i, rect) in enumerate(rects):
        shape = face_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)  # массив точек лица

        if show_video:
            for i in range(36, 48):  # отрисовываем нужные точки глаз
                x, y = shape[i]
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        lefteye = shape[lStart:lEnd]
        righteye = shape[rStart:rEnd]
        leftear = eye_aspect_ratio(lefteye)
        rightear = eye_aspect_ratio(righteye)
        # Среднее соотношение сторон обоих глаз
        ear = (leftear + rightear) / 2.0

        if show_video:
            cv2.putText(image,
                        "EAR: {:.2f}".format(ear),
                        (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2)

        # проверка моргания и прищура
        if ear <= EYE_AR_THRESH:
            duration += 1
            if duration > WAIT_TIME:
                if show_video:
                    cv2.putText(image, "Don't squint!", (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif (ear <= BLINK_THRESH) & (not IS_BLINKING):
                TOTAL_COUNT_BLINKING += 1
                IS_BLINKING = True
        else:
            duration = 0
            IS_BLINKING = False

    if show_video:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(image,
                    "Blinks: {}".format(TOTAL_COUNT_BLINKING),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)
        cv2.imshow('Raw Webcam Feed', image)
    cv2.waitKey(10)
    return ear


'''
ВЫБОРОЧНЫЙ ЗАПУСК ФУНКЦИЙ
Вход:
    scrn_prxmt_trckng - True/False (нужно ли запускать функцию screen_proximity_tracking)
    trckng_eye_shldrs - True/False (нужно ли запускать функцию tracking_eye_shoulders),
    trckng_blnk_sqnt - True/False (нужно ли запускать функцию tracking_blink_squint),
    file_scrn_prxmt_trckng - путь к файлу для записи результата функции screen_proximity_tracking,
    file_trckng_eye_shldrs - путь к файлу для записи результата функции tracking_eye_shoulders,
    file_trckng_blnk_sqnt - путь к файлу для записи результата функции tracking_blink_squint,
    rtgr_scrn_prxmt_trckng - True/False (нужно ли выводить real-time график в функции screen_proximity_tracking),
    rtgr_trckng_eye_shldrs - True/False (нужно ли выводить real-time график в функции tracking_eye_shoulders),
    video_scrn_prxmt_trckng - True/False (нужно ли выводить видео в функции screen_proximity_tracking),
    video_trckng_eye_shldrs - True/False (нужно ли выводить видео в функци tracking_eye_shoulders),
    video_trckng_blnk_sqnt - True/False (нужно ли выводить видео в функци tracking_blink_squint),
    EYE_AR_THRESH=0.25, - пороговое значение прищуривания (для tracking_blink_squint)
    BLINK_THRESH=0.19 - пороговое значение моргания (для tracking_blink_squint),
    WAIT_TIME=2 - количество итераций цикла, в которых должны выполняться условия (для tracking_blink_squint)
Возврат: -
'''


def option_func(scrn_prxmt_trckng=False,
                trckng_eye_shldrs=False,
                trckng_blnk_sqnt=False,
                file_scrn_prxmt_trckng='distance_between_eye_points.txt',
                file_trckng_eye_shldrs='distance_between_eye_shoulders.txt',
                file_trckng_blnk_sqnt='ear.txt',
                rtgr_scrn_prxmt_trckng=False,
                rtgr_trckng_eye_shldrs=False,
                video_scrn_prxmt_trckng=False,
                video_trckng_eye_shldrs=False,
                video_trckng_blnk_sqnt=False,
                EYE_AR_THRESH=0.25,
                BLINK_THRESH=0.18,
                WAIT_TIME=2):
    # очистим все файлы перед записью
    open(file_scrn_prxmt_trckng, 'w').close()
    open(file_trckng_eye_shldrs, 'w').close()
    open(file_trckng_blnk_sqnt, 'w').close()

    cap = cv2.VideoCapture(0)  # захват камеры
    pl_scrn_prxmt_trckng = pl_trckng_eye_shldrs = None

    if rtgr_scrn_prxmt_trckng:
        pl_scrn_prxmt_trckng = Plotter(400, 1000)
    if rtgr_trckng_eye_shldrs:
        pl_trckng_eye_shldrs = Plotter(400, 1000)
    while cap.isOpened():
        if scrn_prxmt_trckng:
            length_line_eyes = screen_proximity_tracking(cap=cap,
                                                         p=pl_scrn_prxmt_trckng,
                                                         show_rtgr=rtgr_scrn_prxmt_trckng,
                                                         show_video=video_scrn_prxmt_trckng)
            # добавим значение в файл
            write_elem_to_file(length_line_eyes, file_scrn_prxmt_trckng)
        if trckng_eye_shldrs:
            length_lines_eyes_sh = tracking_eye_shoulders(cap=cap,
                                                          p=pl_trckng_eye_shldrs,
                                                          show_rtgr=rtgr_trckng_eye_shldrs,
                                                          show_video=video_trckng_eye_shldrs)
            # добавим значение в файл
            write_elem_to_file(length_lines_eyes_sh, file_trckng_eye_shldrs)
        if trckng_blnk_sqnt:
            ear = tracking_blink_squint(cap=cap,
                                        EYE_AR_THRESH=EYE_AR_THRESH,
                                        BLINK_THRESH=BLINK_THRESH,
                                        WAIT_TIME=WAIT_TIME,
                                        show_video=video_trckng_blnk_sqnt)
            # добавим значение в файл
            write_elem_to_file(ear, file_trckng_blnk_sqnt)

        if keyboard.is_pressed('Ctrl + Q'):  # Остановить цикл
            cap.release()
            cv2.destroyAllWindows()
            break


# Тестовый запуск
if __name__ == '__main__':
    #Если вывзвать одну функцию (расстояние между глазами) + реал тайм график + видео
    #option_func(scrn_prxmt_trckng=True, rtgr_scrn_prxmt_trckng=True, video_scrn_prxmt_trckng=True)

    #запуск двух функций + их реал тайм графики + их видео
    #option_func(scrn_prxmt_trckng=True, rtgr_scrn_prxmt_trckng=True, video_scrn_prxmt_trckng=True
    #            ,trckng_eye_shldrs=True, rtgr_trckng_eye_shldrs=True, video_trckng_eye_shldrs=True)

    option_func(scrn_prxmt_trckng=True, trckng_eye_shldrs=True, trckng_blnk_sqnt=True)

    #считывание из файла и построение графика по данным файла
    # eye_distance = get_list_from_file('distance_between_eye_points.txt')
    # print(eye_distance)
    # plt.plot(eye_distance)
    # plt.title('График изменения расстояния между глазами')
    # plt.xlabel('Время')
    # plt.ylabel('Значение расстояния')
    # plt.show()