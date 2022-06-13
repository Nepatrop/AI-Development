import face_recognition
import cv2
import streamlit as st
import numpy as np
from PIL import Imagecat

# update

st.title("Распознавание лица")
st.write("**Обученная нейросеть на основе библиотеки dlib**")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("Выберите метод распознавания лица:")
st.sidebar.write("")
activities = ["С фотографии", "В режиме реального времени"]
choice = st.sidebar.selectbox("select an option", activities)

video_capture = cv2.VideoCapture(0)

# DataSet Faces photos
elonM_F1 = face_recognition.load_image_file("DataSet Faces/1. Elon Musk 1.jpg")
elonM_F1_encoding = face_recognition.face_encodings(elonM_F1)[0]

elonM_F2 = face_recognition.load_image_file("DataSet Faces/2. Elon Musk 2.jpg")
elonM_F2_encoding = face_recognition.face_encodings(elonM_F2)[0]

elonM_F3 = face_recognition.load_image_file("DataSet Faces/3. Elon Musk 3.jpg")
elonM_F3_encoding = face_recognition.face_encodings(elonM_F3)[0]

diKap_F1 = face_recognition.load_image_file("DataSet Faces/4. Di Kaprio 1.jpg")
diKap_F1_encoding = face_recognition.face_encodings(diKap_F1)[0]

diKap_F2 = face_recognition.load_image_file("DataSet Faces/5. Di Kaprio 2.jpg")
diKap_F2_encoding = face_recognition.face_encodings(diKap_F2)[0]

robertD_F1 = face_recognition.load_image_file(
    "DataSet Faces/6. Robert Downey-J 1.jpeg")
robertD_F1_encoding = face_recognition.face_encodings(robertD_F1)[0]

robertD_F2 = face_recognition.load_image_file(
    "DataSet Faces/7. Robert Downey-J 2.jpg")
robertD_F2_encoding = face_recognition.face_encodings(robertD_F2)[0]

sozykinA_F1 = face_recognition.load_image_file(
    "DataSet Faces/8. Sozykin Andrei Vladimirovich 1.png")
sozykinA_F1_encoding = face_recognition.face_encodings(sozykinA_F1)[0]

obabkov_F1 = face_recognition.load_image_file(
    "DataSet Faces/9. Ilia Nikolaevich Obabkov 1.png")
obabkov_F1_encoding = face_recognition.face_encodings(obabkov_F1)[0]

obabkov_F2 = face_recognition.load_image_file(
    "DataSet Faces/10. Ilia Nikolaevich Obabkov 2.png")
obabkov_F2_encoding = face_recognition.face_encodings(obabkov_F2)[0]

obabkov_F3 = face_recognition.load_image_file(
    "DataSet Faces/11. Ilia Nikolaevich Obabkov 3.png")
obabkov_F3_encoding = face_recognition.face_encodings(obabkov_F3)[0]

shadrinD_F = face_recognition.load_image_file(
    "DataSet Faces/12. Shadrin Denis Borisovich.png")
shadrinD_F_encoding = face_recognition.face_encodings(shadrinD_F)[0]

akulovD_F1 = face_recognition.load_image_file(
    "DataSet Faces/13. Akulov Danila 1.jpg")
akulovD_F1_encoding = face_recognition.face_encodings(akulovD_F1)[0]

akulovD_F2 = face_recognition.load_image_file(
    "DataSet Faces/14. Akulov Danila 2.jpg")
akulovD_F2_encoding = face_recognition.face_encodings(akulovD_F2)[0]

malevaniyA_F1 = face_recognition.load_image_file(
    "DataSet Faces/15. Malevaniy Artem 1.jpeg")
malevaniyA_F1_encoding = face_recognition.face_encodings(malevaniyA_F1)[0]

malevaniyA_F2 = face_recognition.load_image_file(
    "DataSet Faces/16. Malevaniy Artem 2.jpeg")
malevaniyA_F2_encoding = face_recognition.face_encodings(malevaniyA_F2)[0]

malevaniyA_F3 = face_recognition.load_image_file(
    "DataSet Faces/17. Malevaniy Artem 3.jpeg")
malevaniyA_F3_encoding = face_recognition.face_encodings(malevaniyA_F3)[0]

sergeevaM_F1 = face_recognition.load_image_file(
    "DataSet Faces/18. Sergeeva Maria 1.jpg")
sergeevaM_F1_encoding = face_recognition.face_encodings(sergeevaM_F1)[0]

sergeevaM_F2 = face_recognition.load_image_file(
    "DataSet Faces/19. Sergeeva Maria 2.jpg")
sergeevaM_F2_encoding = face_recognition.face_encodings(sergeevaM_F2)[0]

# DataSet Masks photos
elonM_M1 = face_recognition.load_image_file("DataSet Masks/1. Elon Musk 1.png")
elonM_M1_encoding = face_recognition.face_encodings(elonM_M1)[0]

elonM_M2 = face_recognition.load_image_file("DataSet Masks/2. Elon Musk 2.jpg")
elonM_M2_encoding = face_recognition.face_encodings(elonM_M2)[0]

elonM_M3 = face_recognition.load_image_file("DataSet Masks/3. Elon Musk 3.jpg")
elonM_M3_encoding = face_recognition.face_encodings(elonM_M3)[0]

LeoDK_M1 = face_recognition.load_image_file("DataSet Masks/4. Di Kaprio 1.jpg")
LeoDK_M1_encoding = face_recognition.face_encodings(LeoDK_M1)[0]

LeoDK_M2 = face_recognition.load_image_file("DataSet Masks/5. Di Kaprio 2.jpg")
LeoDK_M2_encoding = face_recognition.face_encodings(LeoDK_M2)[0]

robertD_M1 = face_recognition.load_image_file(
    "DataSet Masks/6. Robert Downey-J 1.jpg")
robertD_M1_encoding = face_recognition.face_encodings(robertD_M1)[0]

robertD_M2 = face_recognition.load_image_file(
    "DataSet Masks/7. Robert Downey-J 2.jpg")
robertD_M2_encoding = face_recognition.face_encodings(robertD_M2)[0]

sozykinA_M1 = face_recognition.load_image_file(
    "DataSet Masks/8. Andrei Vladimirovich Sozykin 1.jpg")
sozykinA_M1_encoding = face_recognition.face_encodings(sozykinA_M1)[0]

obabkovI_M1 = face_recognition.load_image_file(
    "DataSet Masks/9. Ilia Nikolaevich Obabkov 1.jpg")
obabkovI_M1_encoding = face_recognition.face_encodings(obabkovI_M1)[0]

obabkovI_M2 = face_recognition.load_image_file(
    "DataSet Masks/10. Ilia Nikolaevich Obabkov 2.jpg")
obabkovI_M2_encoding = face_recognition.face_encodings(obabkovI_M2)[0]

obabkovI_M3 = face_recognition.load_image_file(
    "DataSet Masks/11. Ilia Nikolaevich Obabkov 3.jpg")
obabkovI_M3_encoding = face_recognition.face_encodings(obabkovI_M3)[0]

# shadrinD_M1 = face_recognition.load_image_file(
#     "DataSet Masks/12. Denis Borisovich Shadrin  1.jpg")
# shadrinD_M1_encoding = face_recognition.face_encodings(shadrinD_M1)[0]

akulovD_M1 = face_recognition.load_image_file(
    "DataSet Masks/13. Akulov Danila 1.jpg")
akulovD_M1_encoding = face_recognition.face_encodings(akulovD_M1)[0]

akulovD_M2 = face_recognition.load_image_file(
    "DataSet Masks/14. Akulov Danila 2.jpg")
akulovD_M2_encoding = face_recognition.face_encodings(akulovD_M2)[0]

# malevaniyA_M1 = face_recognition.load_image_file("DataSet Masks/17. Malevaniy Artem 3.jpeg")
# malevaniyA_M1_encoding = face_recognition.face_encodings(malevaniyA_M1)[0]

# malevaniyA_M2 = face_recognition.load_image_file("DataSet Masks/16. Malevaniy Artem 2.jpeg")
# malevaniyA_M2_encoding = face_recognition.face_encodings(malevaniyA_M2)[0]

sergeevaM_M1 = face_recognition.load_image_file(
    "DataSet Masks/18. Sergeeva Maria 1.jpg")
sergeevaM_M1_encoding = face_recognition.face_encodings(sergeevaM_M1)[0]

sergeevaM_M2 = face_recognition.load_image_file(
    "DataSet Masks/19. Sergeeva Maria 2.jpg")
sergeevaM_M2_encoding = face_recognition.face_encodings(sergeevaM_M2)[0]

# Create arrays of known face encodings and their names
known_face_encodings = \
    [
        # Faces Encoding arr
        elonM_F1_encoding,
        elonM_F2_encoding,
        elonM_F3_encoding,
        diKap_F1_encoding,
        diKap_F2_encoding,
        robertD_F1_encoding,
        robertD_F2_encoding,

        sozykinA_F1_encoding,
        obabkov_F1_encoding,
        obabkov_F2_encoding,
        obabkov_F3_encoding,
        shadrinD_F_encoding,

        akulovD_F1_encoding,
        akulovD_F2_encoding,
        malevaniyA_F1_encoding,
        malevaniyA_F2_encoding,
        malevaniyA_F3_encoding,
        sergeevaM_F1_encoding,
        sergeevaM_F2_encoding,

        # Masks Encoding arr
        elonM_M1_encoding,
        elonM_M2_encoding,
        elonM_M3_encoding,
        LeoDK_M1_encoding,
        LeoDK_M2_encoding,
        robertD_M1_encoding,
        robertD_M2_encoding,

        sozykinA_M1_encoding,
        obabkovI_M1_encoding,
        obabkovI_M2_encoding,
        obabkovI_M3_encoding,
        # shadrinD_M1_encoding,

        akulovD_M1_encoding,
        akulovD_M2_encoding,
        # malevaniyA_M1_encoding,
        # malevaniyA_M2_encoding,
        sergeevaM_M1_encoding,
        sergeevaM_M2_encoding
    ]

known_face_names = \
    [
        # Faces names Encoding arr
        "Elon Musk",
        "Elon Musk",
        "Elon Musk",
        "Leonardo",
        "Leonardo",
        "Robert D.",
        "Robert D.",

        "Sozykin A. V.",
        "Obabkov I. N.",
        "Obabkov I. N.",
        "Obabkov I. N.",
        "Shadrin D. B.",

        "Akulov D.",
        "Akulov D.",
        "Malevannyi A.",
        "Malevannyi A.",
        "Malevannyi A.",
        "Sergeeva M.",
        "Sergeeva M.",

        # Masks names Encoding arr
        "Elon Musk",
        "Elon Musk",
        "Elon Musk",
        "Leonardo",
        "Leonardo",
        "Robert D.",
        "Robert D.",

        "Sozykin A. V.",
        "Obabkov I. N.",
        "Obabkov I. N.",
        "Obabkov I. N.",
        # "Shadrin D. B.",

        "Akulov D.",
        "Akulov D.",
        # "Malevannyi A.",
        # "Malevannyi A.",
        "Sergeeva M.",
        "Sergeeva M."
    ]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

FRAME_WINDOW = st.image([])

if choice == "В режиме реального времени":
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

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
        # cv2.imshow('Video', frame)
        FRAME_WINDOW.image(frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if choice == "С фотографии":

    image_file = st.file_uploader(
        "Загрузите фотографию", type=['jpeg', 'png', 'jpg', 'webp'])

    if image_file:

        image = Image.open(image_file)

        if st.button("Process"):
            while True:
                # Grab a single frame of video
                frame = np.array(requirements.txtimage.convert('RGB'))
                # Resize frame of video to 1/4 size for faster face recognition processing
                # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                # rgb_small_frame = small_frame[:, :, ::-1]

                # Only process every other frame of video to save time
                # if process_this_frame:
                    # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

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
                # cv2.imshow('Video', frame)
                FRAME_WINDOW.image(frame)


                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
