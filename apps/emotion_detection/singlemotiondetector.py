# Python imports
import warnings
from os.path import abspath, join

# external imports
import cv2
import imutils
import numpy as np
from deepface import DeepFace
from keras.models import load_model
from keras.preprocessing import image

warnings.filterwarnings("ignore")


class SingleMotionDetector:
    def __init__(self, img, emotion, age, race, gender) -> None:
        self.face_haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.media_dir = "static/media/processed_image"

        self.img = img
        self.emotion = emotion
        self.age = age
        self.race = race
        self.gender = gender

        self.action = list()

    def __call__(self, *args, **kwds):
        processed_img = self.deep_face_detector(self.img)
        return processed_img

    # def image_processer(self, test_img):
        # print("*" * 20, test_img, "*" * 20)
        # readed_img = cv2.imread(test_img)
        # result = DeepFace.analyze(
        #     test_img,
        #     actions=["emotion", "age", "race", "gender"],
        #     enforce_detection=False,
        # )
        # gray_img = cv2.cvtColor(readed_img, cv2.COLOR_BGR2RGB)

        # print("*" * 20, "*" * 20)
        # text = (
        #     result["age"]
        #     + " years old "
        #     + result["dominant_race"]
        #     + " "
        #     + result["dominant_emotion"]
        #     + " "
        #     + result["gender"]
        # )
        # print("*" * 20, "*" * 20)

        # face_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        # for (x, y, w, h) in face_detected:
        #     cv2.rectangle(readed_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
        #     roi_gray = gray_img[
        #         y : y + w, x : x + h
        #     ]  # cropping region of interest i.e. face area from  image
        #     roi_gray = cv2.resize(roi_gray, (224, 224))
        #     img_pixels = image.img_to_array(roi_gray)
        #     img_pixels = np.expand_dims(img_pixels, axis=0)
        #     img_pixels /= 255

        #     predictions = self.model.predict(img_pixels)

        #     min_pred = predictions[0]

        #     # find max indexed array
        #     max_index = np.argmax(min_pred)

        #     emotions = (
        #         "angry",
        #         "disgust",
        #         "fear",
        #         "surprise",
        #         "happy",
        #         "sad",
        #         "neutral",
        #     )

        #     predicted_emotion = emotions[max_index]

        #     cv2.putText(
        #         readed_img,
        #         text,
        #         (int(x), int(y)),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (0, 88, 255),
        #         1,
        #     )

        # # resize the width(1000) and the height(700)
        # resized_img = cv2.resize(readed_img, (1000, 700))

        # new_image = f"{join(self.media_dir, test_img.split('/')[-1].split('.')[0])}.jpg"

        # # save the image to the media directory
        # cv2.imwrite(new_image, resized_img)

        # return new_image

    def deep_face_detector(self, image):
        readed_img = cv2.imread(image)

        actions_list = self.get_action_list()

        print(actions_list)

        result = DeepFace.analyze(
            readed_img,
            actions=actions_list,
            enforce_detection=False,
        )

        print(result)

        print(result["dominant_emotion"])

        gray_img = cv2.cvtColor(readed_img, cv2.COLOR_BGR2RGB)

        face_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        text = (
            str(result["age"]) if "age" in actions_list else ""
            + " years old " if "age" in actions_list else ""
            + str(result["dominant_race"]) if "dominant_race" in actions_list else ""
            + " " if "dominant_race" in actions_list else ""
            + str(result["dominant_emotion"]) if "dominant_emotion" in actions_list else ""
            + " " if "dominant_emotion" in actions_list else ""
            + str(result["gender"]) if "gender" in actions_list else ""
        )


        print("Text: ", text)

        for (x, y, w, h) in face_detected:
            cv2.rectangle(readed_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

            cv2.putText(
                readed_img,
                text,
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 88, 255),
                1,
            )

        # resize the width(1000) and the height(700)
        resized_img = cv2.resize(readed_img, (1000, 700))

        new_image = f"{join(self.media_dir, image.split('/')[-1].split('.')[0])}.jpg"

        # save the image to the media directory
        cv2.imwrite(new_image, resized_img)

        return new_image

    def get_action_list(self):
        self.action.append("emotion" if self.emotion else None)
        self.action.append("race" if self.race else None)
        self.action.append("gender" if self.gender else None)
        self.action.append("age" if self.age else None)

        return [action for action in self.action if action is not None]
