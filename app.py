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
        # Page configuration
        st.set_page_config(page_title='Visual Math Interpreter', layout="wide")

        # Page header transparent color and removes top padding
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

        # Title and position
        st.markdown(f'<h1 style="text-align: center;">Visual Math Interpreter</h1>',
                    unsafe_allow_html=True)
        # Instructions for the user
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
        # Load the environment variables for the secret API key
        load_dotenv()

        # Initialize webcam capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 650)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 130)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                         min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mp_drawing = mp.solutions.drawing_utils

        # Canvas for drawing
        self.imgCanvas = np.zeros((650, 900, 3), dtype=np.uint8)

        # Previous points for drawing
        self.p1, self.p2 = 0, 0

        # List to hold finger states
        self.fingers = []

        # Initialize the image attribute
        self.img = None

        # Flag to prevent multiple analyses
        self.analyze_triggered = False

    def process_frame(self):
        # Read video capture
        success, img = self.cap.read()
        if not success:
            return None

        img = cv2.resize(img, (900, 650))
        img = cv2.flip(img, 1)
        return img

    def process_hands(self, img):
        # Convert the image to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Detect hand landmarks
        result = self.hands.process(rgb_img)

        self.landmark_list = []
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                for id, lm in enumerate(hand_lms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmark_list.append([id, cx, cy])
        self.img = img  # Assign the processed image to self.img
        return img

    def identify_fingers(self):
        # Identify which fingers are open (0: closed, 1: open)
        self.fingers = []

        if self.landmark_list:
            # Thumb
            # Check if thumb is open: Compare x coordinates for right hand
            if self.landmark_list[4][1] < self.landmark_list[3][1]:
                self.fingers.append(1)
            else:
                self.fingers.append(0)

            # Fingers
            tip_ids = [8, 12, 16, 20]
            for tip_id in tip_ids:
                if self.landmark_list[tip_id][2] < self.landmark_list[tip_id - 2][2]:
                    self.fingers.append(1)
                else:
                    self.fingers.append(0)

            # Highlight open fingers
            for i in range(len(self.fingers)):
                if self.fingers[i] == 1:
                    finger_tip = (self.landmark_list[(i + 1) * 4][1], self.landmark_list[(i + 1) * 4][2])
                    cv2.circle(img=self.img, center=finger_tip, radius=5, color=(0, 255, 255), thickness=1)

    def handle_drawing_mode(self):
        # Drawing mode: Thumb and Index fingers up
        if sum(self.fingers) == 2 and self.fingers[0] == 1 and self.fingers[1] == 1:
            cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), (0, 255, 255), 5)
            self.p1, self.p2 = cx, cy
        # Disable drawing when Thumb, Index & Middle fingers are up
        elif sum(self.fingers) == 3 and all(finger == 1 for finger in self.fingers[:3]):
            self.p1, self.p2 = 0, 0
        # Erase mode: Thumb and Middle fingers up
        elif sum(self.fingers) == 2 and self.fingers[0] == 1 and self.fingers[2] == 1:
            cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), (0, 0, 0), 15)
            self.p1, self.p2 = cx, cy
        # Reset mode: Thumb and Pinky fingers up
        elif sum(self.fingers) == 2 and self.fingers[0] == 1 and self.fingers[4] == 1:
            self.imgCanvas = np.zeros((650, 900, 3), dtype=np.uint8)
            st.session_state.analyze_triggered = False  # Reset the flag for new drawing
        else:
            self.p1, self.p2 = 0, 0

    def blend_canvas_with_feed(self, img):
        # Blend the canvas with the webcam feed
        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, self.imgCanvas)
        return img

    def analyze_image_with_genai(self):
        # Canvas_BGR Image Convert to RGB Image
        imgCanvas = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2RGB)

        # Numpy Array Convert to PIL Image
        imgCanvas = PIL.Image.fromarray(imgCanvas)

        # Configure the genai Library
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

        # Initialize a Flash Generative Model
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')

        # Input Prompt
        prompt = (
            "Analyze the image and provide the following:\n"
            "* The mathematical equation represented in the image.\n"
            "* The solution to the equation.\n"
            "* A short and sweet explanation of the steps taken to arrive at the solution."
        )

        # Send Request to Model to Generate Content using a Text Prompt and Image
        response = model.generate_content([prompt, imgCanvas])

        # Extract the Text Content of the Model’s Response.
        return response.text

    def main(self):
        # Streamlit layout
        col1, _, col3 = st.columns([0.8, 0.02, 0.18])

        with col1:
            # Placeholder for the video frame
            stframe = st.empty()

        with col3:
            # Placeholder for the result output
            st.markdown(f'<h5 style="text-align:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()

        # Initialize session state for analyze_triggered
        if 'analyze_triggered' not in st.session_state:
            st.session_state.analyze_triggered = False

        while True:
            if not self.cap.isOpened():
                add_vertical_space(5)
                st.markdown(
                    body=(
                        '<h4 style="text-align:center; color:orange;">'
                        'Error: Could not open webcam. Please ensure your webcam is connected and try again.'
                        '</h4>'
                    ),
                    unsafe_allow_html=True
                )
                break

            img = self.process_frame()
            if img is None:
                st.markdown(
                    body=(
                        '<h4 style="text-align:center; color:orange;">'
                        'Error: Failed to read from webcam. Please try again.'
                        '</h4>'
                    ),
                    unsafe_allow_html=True
                )
                break

            img = self.process_hands(img)
            self.identify_fingers()
            self.handle_drawing_mode()
            blended_img = self.blend_canvas_with_feed(img)

            # Display the Output Frame in the Streamlit App
            blended_rgb = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
            stframe.image(blended_rgb, channels="RGB")

            # Check for the specific gesture: Only Index and Middle fingers up
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
                        st.session_state.analyze_triggered = True  # Prevent multiple analyses
                    except Exception as e:
                        result_placeholder.write(f"**Error during analysis:** {e}")

            # To allow Streamlit to handle other events, add a small sleep
            time.sleep(0.1)

            # Break the loop if Streamlit session is stopped
            if not st.session_state.get('run', True):
                break

        # Release the camera and close windows
        self.cap.release()
        cv2.destroyAllWindows()


try:
    # Create an instance of the class
    interpreter = VisualMathInterpreter()

    # Streamlit Configuration Setup
    interpreter.streamlit_config()

    # Run the main method
    interpreter.main()

except Exception as e:
    add_vertical_space(5)
    # Display the Error Message
    st.markdown(
        f'<h5 style="text-align:center;color:orange;">Error: {e}</h5>',
        unsafe_allow_html=True
    )
