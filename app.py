import os
import cv2
import PIL
import numpy as np
import google.generativeai as genai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import mediapipe as mp
from dotenv import load_dotenv
from warnings import filterwarnings
import time

filterwarnings(action='ignore')


class VisualMathInterpreter:
    def streamlit_config(self):
        st.set_page_config(page_title='Visual Math Interpreter', layout="wide")
        page_background_color = """
        <style>
            [data-testid="stHeader"] {
                background: rgba(0,0,0,0);
            }

            .block-container {
                padding-top: 0rem;
            }
        </style>
        """
        st.markdown(page_background_color, unsafe_allow_html=True)
        st.markdown(f'<h1 style="text-align: center;">Visual Math Interpreter</h1>', unsafe_allow_html=True)
        st.markdown(
            """
            <h4 style='text-align: center; color: yellow;'>
                Use your hand to draw on the screen.
                <br>
                - **Thumb + Index Finger**: Draw
                <br>
                - **Thumb + Middle Finger**: Erase
                <br>
                - **Thumb + Pinky Finger**: Reset Canvas
                <br>
                - **Index + Middle Fingers**: Analyze Drawing
            </h4>
            """,
            unsafe_allow_html=True
        )
        add_vertical_space(1)

    def __init__(self):
        load_dotenv()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 650)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                         min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_drawing = mp.solutions.drawing_utils

        self.imgCanvas = np.zeros((650, 900, 3), dtype=np.uint8)
        self.p1, self.p2 = 0, 0
        self.fingers = []
        self.img = None

    def process_frame(self):
        success, img = self.cap.read()
        if not success:
            return None
        img = cv2.resize(img, (900, 650))
        img = cv2.flip(img, 1)
        return img

    def process_hands(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_img)
        self.landmark_list = []
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                for id, lm in enumerate(hand_lms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmark_list.append([id, cx, cy])
        self.img = img
        return img

    def identify_fingers(self):
        self.fingers = []
        if self.landmark_list:
            if self.landmark_list[4][1] < self.landmark_list[3][1]:
                self.fingers.append(1)
            else:
                self.fingers.append(0)
            tip_ids = [8, 12, 16, 20]
            for tip_id in tip_ids:
                if self.landmark_list[tip_id][2] < self.landmark_list[tip_id - 2][2]:
                    self.fingers.append(1)
                else:
                    self.fingers.append(0)

    def handle_drawing_mode(self):
        if sum(self.fingers) == 2 and self.fingers[0] == 1 and self.fingers[1] == 1:
            cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), (0, 255, 255), 5)
            self.p1, self.p2 = cx, cy
        elif sum(self.fingers) == 3 and all(finger == 1 for finger in self.fingers[:3]):
            self.p1, self.p2 = 0, 0
        elif sum(self.fingers) == 2 and self.fingers[0] == 1 and self.fingers[2] == 1:
            cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), (0, 0, 0), 15)
            self.p1, self.p2 = cx, cy
        elif sum(self.fingers) == 2 and self.fingers[0] == 1 and self.fingers[4] == 1:
            self.imgCanvas = np.zeros((650, 900, 3), dtype=np.uint8)
            st.session_state.analyze_triggered = False
        else:
            self.p1, self.p2 = 0, 0

    def blend_canvas_with_feed(self, img):
        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, self.imgCanvas)
        return img

    def analyze_image_with_genai(self, img=None):
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')

        if img is None:
            imgCanvas = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2RGB)
            imgCanvas = PIL.Image.fromarray(imgCanvas)
        else:
            imgCanvas = img

        prompt = (
            "Analyze the image and provide the following:\n"
            "* The mathematical equation represented in the image.\n"
            "* The solution to the equation.\n"
            "* A short and sweet explanation of the steps taken to arrive at the solution."
        )

        response = model.generate_content([prompt, imgCanvas])
        return response.text

    def main(self):
        col1, _, col3 = st.columns([0.8, 0.02, 0.18])

        with col1:
            stframe = st.empty()
            uploaded_file = st.file_uploader("Upload an image or file", type=["jpg", "png"])
            analyze_button = st.button("Analyze Uploaded File")

        with col3:
            st.markdown(f'<h5 style="text-align:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()

        if 'analyze_triggered' not in st.session_state:
            st.session_state.analyze_triggered = False

        if 'upload_analyze_done' not in st.session_state:
            st.session_state.upload_analyze_done = False

        while True:
            if not self.cap.isOpened():
                break

            img = self.process_frame()
            if img is None:
                break

            img = self.process_hands(img)
            self.identify_fingers()
            self.handle_drawing_mode()
            blended_img = self.blend_canvas_with_feed(img)

            blended_rgb = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
            stframe.image(blended_rgb, channels="RGB")

            if (
                sum(self.fingers) == 2 and
                self.fingers[1] == 1 and
                self.fingers[2] == 1 and
                self.fingers[0] == 0 and
                self.fingers[3] == 0 and
                self.fingers[4] == 0 and
                not st.session_state.analyze_triggered
            ):
                with st.spinner('Analyzing your drawing...'):
                    try:
                        result = self.analyze_image_with_genai()
                        result_placeholder.write(f"**Result:** {result}")
                        st.session_state.analyze_triggered = True
                        st.session_state.upload_analyze_done = False
                    except Exception as e:
                        result_placeholder.write(f"**Error during analysis:** {e}")

            if analyze_button and uploaded_file is not None and not st.session_state.upload_analyze_done:
                with st.spinner('Analyzing uploaded image...'):
                    try:
                        uploaded_image = PIL.Image.open(uploaded_file)
                        result = self.analyze_image_with_genai(img=uploaded_image)
                        result_placeholder.write(f"**Result:** {result}")
                        st.session_state.upload_analyze_done = True
                        st.session_state.analyze_triggered = False
                    except Exception as e:
                        result_placeholder.write(f"**Error during analysis:** {e}")

            time.sleep(0.1)

            if not st.session_state.get('run', True):
                break

        self.cap.release()
        cv2.destroyAllWindows()


try:
    interpreter = VisualMathInterpreter()
    interpreter.streamlit_config()
    interpreter.main()

except Exception as e:
    add_vertical_space(5)
    st.markdown(
        f'<h5 style="text-align:center;color:orange;">Error: {e}</h5>',
        unsafe_allow_html=True
    )
