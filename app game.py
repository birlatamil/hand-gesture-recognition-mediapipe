#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier

from utils import calc_landmark_list, pre_process_landmark

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=960)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


'''def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        numbers = []
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                recognized_number = keypoint_classifier_labels[hand_sign_id]
                try:
                    number_value = int(recognized_number)
                    numbers.append(number_value)
                except ValueError:
                    number_value = -1  # fallback if not a digit
                if number_value != -1:
                    if number_value % 2 == 0:
                        result_text = f"{number_value} is Even"
                    else:
                        result_text = f"{number_value} is Odd"

                    cv.putText(debug_image, result_text,
                        (10, 180), cv.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0), 2, cv.LINE_AA)

                # Show detected number
                cv.putText(debug_image, f"Detected Number: {recognized_number}",
                        (10, 140), cv.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3, cv.LINE_AA)

                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

        if len(numbers) == 2:
            total = numbers[0] + numbers[1]
            cv.putText(debug_image, f"Sum: {total}", (10, 220),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv.LINE_AA)
    cap.release()
    cv.destroyAllWindows()'''


# === Game Classes ===
class Player:
    def __init__(self, name):
        self.name = name
        self.choice = ""  # Odd or Even
        self.role = ""    # Bat or Bowl
        self.score = 0

# === Load Models ===
def load_model():
    keypoint_classifier = KeyPointClassifier()
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return hands, keypoint_classifier, keypoint_classifier_labels

# === Detect Number from Gesture ===
'''def get_number(player_name, hands, classifier, labels):
    cap = cv.VideoCapture(0)
    print(f"{player_name}, show your hand (0-5)... Hold steady.")

    detected_number = None
    stable_frames = 0
    previous_number = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed = pre_process_landmark(landmark_list)
                hand_id = classifier(pre_processed)
                try:
                    number = int(labels[hand_id])
                    cv.putText(debug_image, f"Detected: {number}", (10, 60),
                               cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    if number == previous_number:
                        stable_frames += 1
                    else:
                        stable_frames = 0
                        previous_number = number
                    if stable_frames >= 10:
                        detected_number = number
                        break
                except:
                    pass

        cv.putText(debug_image, f"{player_name}, show number", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv.imshow("Detect Number", debug_image)

        if cv.waitKey(1) == 27 or detected_number is not None:
            break

    cap.release()
    cv.destroyAllWindows()
    return detected_number'''


'''def get_two_player_numbers(hands, classifier, labels):
    import time
    cap = cv.VideoCapture(0)
    print("Both players: Show your signs (0-5)... Hold steady!")

    detected_numbers = [None, None]
    stable_frames = [0, 0]
    previous_numbers = [-1, -1]

    start_time = time.time()
    timeout = 30  # seconds to wait before abort

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        detected_hands = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # ✨ Draw skeleton on debug_image
                mp_drawing.draw_landmarks(
                debug_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
            )

                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed = pre_process_landmark(landmark_list)
                hand_id = classifier(pre_processed)
                try:
                    number = int(labels[hand_id])
                except:
                    number = -1

                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                detected_hands.append((hand_label, number))

                cv.putText(debug_image, f"{hand_label}: {number}",
                           (10, 30 if hand_label == "Left" else 60),
                           cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Assign numbers based on hand labels
        for hand_label, number in detected_hands:
            index = 0 if hand_label == "Left" else 1
            if number == previous_numbers[index]:
                stable_frames[index] += 1
            else:
                stable_frames[index] = 0
                previous_numbers[index] = number

            if stable_frames[index] >= 10:
                detected_numbers[index] = number

        cv.putText(debug_image, "Show numbers with both hands", (10, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv.imshow("Two Player Gesture", debug_image)

        # If both hands are detected
        if all(n is not None for n in detected_numbers):
            break

        # Timeout after 10 seconds
        if time.time() - start_time > timeout:
            print("⚠️ Timed out! Show signs quickly next time.")
            break

        if cv.waitKey(1) == 27:  # ESC to abort
            break

    cap.release()
    cv.destroyAllWindows()

    n1 = detected_numbers[0] if detected_numbers[0] is not None else 0
    n2 = detected_numbers[1] if detected_numbers[1] is not None else 0
    print(f"✅ Player 1 (Left): {n1}, Player 2 (Right): {n2}")
    return n1, n2


# ✅ Optimized Hand Detection for Two-Player Gesture Input
# Uses high-resolution, improved smoothing, and detailed debug info
'''




'''def get_two_player_numbers(cap, hands, classifier, labels):
    import time
    print("Both players: Show your signs (0-5)... Hold steady!")

    detected_numbers = [None, None]
    stable_frames = [0, 0]
    previous_numbers = [-1, -1]

    start_time = time.time()
    timeout = 30  # seconds to wait before abort

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        detected_hands = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                )

                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed = pre_process_landmark(landmark_list)
                hand_id = classifier(pre_processed)
                try:
                    number = int(labels[hand_id])
                except:
                    number = -1

                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                detected_hands.append((hand_label, number))

                y_offset = 30 if hand_label == "Left" else 60
                cv.putText(debug_image, f"{hand_label}: {number}",
                           (10, y_offset),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Track stable gestures
        for hand_label, number in detected_hands:
            index = 0 if hand_label == "Left" else 1
            if number == previous_numbers[index]:
                stable_frames[index] += 1
            else:
                stable_frames[index] = 0
                previous_numbers[index] = number

            if stable_frames[index] >= 7:
                detected_numbers[index] = number

        # Overlay info
        cv.putText(debug_image, "Show numbers with both hands", (10, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv.putText(debug_image, f"Left Stable: {stable_frames[0]}", (10, 130),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv.putText(debug_image, f"Right Stable: {stable_frames[1]}", (10, 160),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv.imshow("Two Player Gesture", debug_image)

        if all(n is not None for n in detected_numbers):
            break

        if time.time() - start_time > timeout:
            print("\u26a0\ufe0f Timed out! Show signs quickly next time.")
            break

        if cv.waitKey(1) == 27:
            break

    print(f"\u2705 Player 1 (Left): {detected_numbers[0]}, Player 2 (Right): {detected_numbers[1]}")
    return detected_numbers[0] or 0, detected_numbers[1] or 0


'''

# ✅ Optimized Hand Detection for Two-Player Gesture Input
# Uses high-resolution, improved smoothing, and detailed debug info

def get_two_player_numbers(cap, hands, classifier, labels):
    import time
    print("Both players: Show your signs (0-5)... Hold steady!")

    detected_numbers = [None, None]
    stable_frames = [0, 0]
    previous_numbers = [-1, -1]

    start_time = time.time()
    timeout = 30  # seconds to wait before abort

    waiting_phase = True
    print("✋ Waiting for both players to show '0' (fist) to start...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        detected_hands = {"Left": None, "Right": None}

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                )

                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed = pre_process_landmark(landmark_list)
                hand_id = classifier(pre_processed)
                try:
                    number = int(labels[hand_id])
                except:
                    number = -1

                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                detected_hands[hand_label] = number

                y_offset = 30 if hand_label == "Left" else 60
                cv.putText(debug_image, f"{hand_label}: {number}",
                           (10, y_offset),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if waiting_phase:
            if detected_hands["Left"] == 0 and detected_hands["Right"] == 0:
                print("🎮 Both players ready. Start showing real signs!")
                waiting_phase = False
                stable_frames = [0, 0]
                previous_numbers = [-1, -1]
                time.sleep(1)
            else:
                cv.putText(debug_image, "Show '0' on both hands to begin", (10, 100),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv.imshow("Two Player Gesture", debug_image)
                if time.time() - start_time > timeout:
                    print("⚠️ Timed out! Start signal not detected.")
                    break
                if cv.waitKey(1) == 27:
                    break
                continue

        for idx, hand_label in enumerate(["Left", "Right"]):
            number = detected_hands[hand_label]
            if number == previous_numbers[idx]:
                stable_frames[idx] += 1
            else:
                stable_frames[idx] = 0
                previous_numbers[idx] = number

            if stable_frames[idx] >= 7:
                detected_numbers[idx] = number

        cv.putText(debug_image, "Show numbers with both hands", (10, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv.putText(debug_image, f"Left Stable: {stable_frames[0]}", (10, 130),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv.putText(debug_image, f"Right Stable: {stable_frames[1]}", (10, 160),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv.imshow("Two Player Gesture", debug_image)

        if all(n is not None for n in detected_numbers):
            break

        if time.time() - start_time > timeout:
            print("⚠️ Timed out! Show signs quickly next time.")
            break

        if cv.waitKey(1) == 27:
            break

    print(f"✅ Player 1 (Left): {detected_numbers[0]}, Player 2 (Right): {detected_numbers[1]}")
    return detected_numbers[0] or 0, detected_numbers[1] or 0



# === Full Game Flow ===
'''def odd_even_game():
    hands, classifier, labels = load_model()
    p1 = Player("Player 1")
    p2 = Player("Player 2")

    print("\n==== ODD-EVEN TOSS ====")
    p1.choice = input("Player 1, choose Odd or Even: ").strip().lower()
    p2.choice = "even" if p1.choice == "odd" else "odd"

    n1, n2 = get_two_player_numbers(cap, hands, classifier, labels)


    total = n1 + n2
    print(f"Player 1: {n1}, Player 2: {n2}, Total: {total} => {'Even' if total % 2 == 0 else 'Odd'}")

    toss_winner = p1 if (total % 2 == 0 and p1.choice == "even") or (total % 2 != 0 and p1.choice == "odd") else p2
    toss_loser = p2 if toss_winner == p1 else p1

    toss_winner.role = input(f"{toss_winner.name}, choose role (bat/bowl): ").strip().lower()
    toss_loser.role = "bowl" if toss_winner.role == "bat" else "bat"

    print(f"\n{toss_winner.name} is {toss_winner.role}, {toss_loser.name} is {toss_loser.role}")

    # Set innings
    batter = toss_winner if toss_winner.role == "bat" else toss_loser
    bowler = toss_loser if batter == toss_winner else toss_winner

    print(f"\n==== First Innings: {batter.name} Batting ====")
    while True:
        bat, bowl = get_two_player_numbers(cap, hands, classifier, labels)
        print(f"{batter.name}: {bat}, {bowler.name}: {bowl}")
        if bat == bowl:
            print(f"{batter.name} is OUT!")
            break
        else:
            batter.score += bat
            print(f"Score: {batter.score}")

    target = batter.score + 1
    print(f"\nTarget for {bowler.name}: {target}")

    # Second innings
    batter, bowler = bowler, batter
    batter.score = 0

    print(f"\n==== Second Innings: {batter.name} Batting ====")
    while True:
        bat, bowl = get_two_player_numbers(cap, hands, classifier, labels)
        print(f"{batter.name}: {bat}, {bowler.name}: {bowl}")
        if bat == bowl:
            print(f"{batter.name} is OUT!")
            break
        else:
            batter.score += bat
            print(f"Score: {batter.score}")
            if batter.score >= target:
                print(f"{batter.name} has chased the target!")
                break

    print("\n==== RESULT ====")
    if batter.score >= target:
        print(f"{batter.name} wins!")
    else:
        print(f"{bowler.name} wins!")
'''

def odd_even_game(cap):
    hands, classifier, labels = load_model()
    p1 = Player("Player 1")
    p2 = Player("Player 2")

    print("\n==== ODD-EVEN TOSS ====")
    p1.choice = input("Player 1, choose Odd or Even: ").strip().lower()
    p2.choice = "even" if p1.choice == "odd" else "odd"

    n1, n2 = get_two_player_numbers(cap, hands, classifier, labels)

    total = n1 + n2
    print(f"Player 1: {n1}, Player 2: {n2}, Total: {total} => {'Even' if total % 2 == 0 else 'Odd'}")

    toss_winner = p1 if (total % 2 == 0 and p1.choice == "even") or (total % 2 != 0 and p1.choice == "odd") else p2
    toss_loser = p2 if toss_winner == p1 else p1

    toss_winner.role = input(f"{toss_winner.name}, choose role (bat/bowl): ").strip().lower()
    toss_loser.role = "bowl" if toss_winner.role == "bat" else "bat"

    print(f"\n{toss_winner.name} is {toss_winner.role}, {toss_loser.name} is {toss_loser.role}")

    # Set innings
    batter = toss_winner if toss_winner.role == "bat" else toss_loser
    bowler = toss_loser if batter == toss_winner else toss_winner

    print(f"\n==== First Innings: {batter.name} Batting ====")
    while True:
        bat, bowl = get_two_player_numbers(cap, hands, classifier, labels)
        print(f"{batter.name}: {bat}, {bowler.name}: {bowl}")
        if bat == bowl:
            print(f"{batter.name} is OUT!")
            break
        else:
            batter.score += bat
            print(f"Score: {batter.score}")

    target = batter.score + 1
    print(f"\nTarget for {bowler.name}: {target}")

    # Second innings
    batter, bowler = bowler, batter
    batter.score = 0

    print(f"\n==== Second Innings: {batter.name} Batting ====")
    while True:
        bat, bowl = get_two_player_numbers(cap, hands, classifier, labels)
        print(f"{batter.name}: {bat}, {bowler.name}: {bowl}")
        if bat == bowl:
            print(f"{batter.name} is OUT!")
            break
        else:
            batter.score += bat
            print(f"Score: {batter.score}")
            if batter.score >= target:
                print(f"{batter.name} has chased the target!")
                break

    print("\n==== RESULT ====")
    if batter.score >= target:
        print(f"{batter.name} wins!")
    else:
        print(f"{bowler.name} wins!")


if __name__ == "__main__":
    cap = cv.VideoCapture(0)  # Initialize the camera once
    odd_even_game(cap)        # Pass it to the game
    cap.release()             # Release after game finishes
    cv.destroyAllWindows()



def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


