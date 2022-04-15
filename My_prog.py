import dlib
import cv2
from imutils import face_utils
import numpy as np

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape) #массив точек лица

        for i in range (len(shape)):
            x, y = shape[i]
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # нарисуем центр левого глаза глаза
        x1, y1 = shape[36]
        x2, y2 = shape[39]
        x_center = round((x1 + x2) / 2)
        y_center = round((y1 + y2) / 2)
        cv2.circle(image, (x_center, y_center), 2, (0, 0, 255), -1)

        # нарисуем центр правого глаза глаза
        x1, y1 = shape[42]
        x2, y2 = shape[45]
        x_center = round((x1 + x2) / 2)
        y_center = round((y1 + y2) / 2)
        cv2.circle(image, (x_center, y_center), 2, (0, 0, 255), -1)

    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()