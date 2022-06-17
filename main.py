import face_recognition
import cv2
import streamlit as st
import numpy as np
from PIL import Image

st.title("AI FACE RECOGNIZER")
st.write("**Обученная нейросеть на основе библиотеки dlib, способная распознавать лица людей в медицинской маске и без**")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("Выберите метод распознавания лица:")
instruction = "Инструкция"
photoDetection = "С фотографии"
liveCamDetection = "В режиме реального времени"
activities = [instruction, photoDetection, liveCamDetection]
choice = st.sidebar.selectbox("", activities)

if choice == instruction:
    st.write("На этой странице вы можете увидеть инструкцию по использованию нашего веб-приложения:")
    st.write("В левой части окна вы можете выбрать один из двух методов распознавания лица: с фотографии и в режиме реального времени (в этом случае вам придется включить фронтальную камеру).")
    st.write("В случае выбора распознавания по фотографии вам откроется панель, куда нужно загрузить файл в формате jpg, jpeg, png или webp. После чего просто дождаться, пока алгоритм закончит работу и выведет результат.")
    st.write("В случае выбора распознавания в режиме реального времени появится окно, в которое будет выведено изображение с веб-камеры. Алгоритм определяет лица в зоне видимости камеры мгновенно.")
    st.write("Поскольку данный алгоритм используется для контроля доступа (к устройству, системе, для входа куда-либо), то при удачной попытке распознавания лица вам будет разрешен доступ к абсрактной системе. В противном случае в доступе будет отказано.")
    st.write("")
    st.write("")
    st.write("")
    st.caption("**Сделано командой AI CREATORS**")



# DataSet Faces photos
obabkov_F1 = face_recognition.load_image_file(
    "DataSet Masks/9. Ilia Nikolaevich Obabkov 1.jpg")
obabkov_F1_encoding = face_recognition.face_encodings(obabkov_F1)[0]

# Create arrays of known face encodings and their names
known_face_encodings = \
    [
        # Faces Encoding arr
        obabkov_F1_encoding
    ]

known_face_names = \
    [
        # Faces names Encoding arr
        "Obabkov I. N."
    ]



face_locations = []
face_encodings = []
face_names = []
process_this_frame = True



FRAME_WINDOW = st.image([])



if choice == liveCamDetection:
    video_capture = cv2.VideoCapture(CAP_V4L2)
    successCheck = True
    errorCheck = True
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            continue
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(fix_frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

                if name != "Unknown" and successCheck:
                    st.success("Доступ к системе разрешен")
                    successCheck = False
                elif name == "Unknown" and errorCheck:
                    st.error("В доступе отказано")
                    errorCheck = False
        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        FRAME_WINDOW.image(frame)
    # Release handle to the webcam
    video_capture.release()



if choice == photoDetection:

    image_file = st.file_uploader(
        "Загрузите фотографию", type=['jpeg', 'png', 'jpg', 'webp'])

    if image_file:
        image = Image.open(image_file)
        successCheck = True
        errorCheck = True
        while True:
            # Grab a single frame of video
            frame = np.array(image.convert('RGB'))
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            # if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

                if name != "Unknown" and successCheck:
                    st.success("Доступ к системе разрешен")
                    successCheck = False
                elif name == "Unknown" and errorCheck:
                    st.error("В доступе отказано")
                    errorCheck = False

            #process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            FRAME_WINDOW.image(frame)