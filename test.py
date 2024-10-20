import cv2
import numpy as np
import mediapipe as mp

class VisualMathInterpreter:

    def __init__(self):
        # Initialize webcam capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 650)

        # Initialize MediaPipe hands solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                                         min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils

        # Canvas for drawing
        self.imgCanvas = np.zeros((650, 900, 3), np.uint8)
        self.p1, self.p2 = 0, 0
        self.fingers = []

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
        return img

    def identify_fingers(self):
        # Identify which fingers are open (0: closed, 1: open)
        self.fingers = []

        if self.landmark_list != []:
            for id in [4, 8, 12, 16, 20]:  # thumb, index, middle, ring, pinky
                if id != 4:  # For all fingers except thumb
                    if self.landmark_list[id][2] < self.landmark_list[id - 2][2]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)
                else:  # For thumb
                    if self.landmark_list[id][1] < self.landmark_list[id - 2][1]:
                        self.fingers.append(1)
                    else:
                        self.fingers.append(0)

    def handle_drawing_mode(self):
        # Drawing mode: Thumb and Index fingers up
        if sum(self.fingers) == 2 and self.fingers[0] == self.fingers[1] == 1:
            cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), (0, 255, 255), 5)
            self.p1, self.p2 = cx, cy

        # Erase mode: Thumb and middle fingers up
        elif sum(self.fingers) == 2 and self.fingers[0] == self.fingers[2] == 1:
            cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]
            if self.p1 == 0 and self.p2 == 0:
                self.p1, self.p2 = cx, cy
            cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), (0, 0, 0), 15)
            self.p1, self.p2 = cx, cy

        # Reset mode: Thumb and pinky fingers up
        elif sum(self.fingers) == 2 and self.fingers[0] == self.fingers[4] == 1:
            self.imgCanvas = np.zeros((650, 900, 3), np.uint8)

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

    def main(self):
        while True:
            img = self.process_frame()
            if img is None:
                break

            img = self.process_hands(img)
            self.identify_fingers()
            self.handle_drawing_mode()
            img = self.blend_canvas_with_feed(img)

            cv2.imshow("Visual Math Interpreter", img)
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Run the Visual Math Interpreter
if __name__ == "__main__":
    try:
        calc = VisualMathInterpreter()
        calc.main()
    except Exception as e:
        print(f"Error: {e}")  