import numpy as np
import pyautogui
import time

class GestureRecognizer:
    def __init__(self):
        self.previous_gesture = None
        self.gesture_threshold = 0.15
        self.last_action_time = 0
        self.cooldown = 0.3
        self.previous_wrist_positions = {"left": None, "right": None}
        
    def get_landmark_coords(self, landmarks, index):
        """Extract x, y, z coordinates of a landmark"""
        return landmarks[index].x, landmarks[index].y, landmarks[index].z
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + 
                      (point1[1] - point2[1])**2 + 
                      (point1[2] - point2[2])**2)
    
    def detect_punch_left(self, landmarks):
        """Detect left punch - left wrist extends forward from shoulder"""
        left_shoulder = self.get_landmark_coords(landmarks, 11)
        left_elbow = self.get_landmark_coords(landmarks, 13)
        left_wrist = self.get_landmark_coords(landmarks, 15)
        right_wrist = self.get_landmark_coords(landmarks, 16)

        arm_extension = left_wrist[2] < left_shoulder[2] - 0.25
        elbow_extended = self.calculate_distance(left_shoulder, left_wrist) > 0.45
        
        wrist_forward = left_wrist[2] < left_elbow[2]
        
        right_hand_relaxed = right_wrist[2] > left_wrist[2] + 0.1
        

        wrist_at_punch_height = abs(left_wrist[1] - left_shoulder[1]) < 0.2
        
        is_fast = False
        if self.previous_wrist_positions["left"]:
            movement = abs(left_wrist[2] - self.previous_wrist_positions["left"][2])
            is_fast = movement > 0.03
        
        self.previous_wrist_positions["left"] = left_wrist # type: ignore
        
        return (arm_extension and elbow_extended and wrist_forward and 
                right_hand_relaxed and wrist_at_punch_height and is_fast)
    
    def detect_punch_right(self, landmarks):
        """Detect right punch"""
        right_shoulder = self.get_landmark_coords(landmarks, 12)
        right_elbow = self.get_landmark_coords(landmarks, 14)
        right_wrist = self.get_landmark_coords(landmarks, 16)
        left_wrist = self.get_landmark_coords(landmarks, 15)
        
        arm_extension = right_wrist[2] < right_shoulder[2] - 0.25  # Stricter threshold
        elbow_extended = self.calculate_distance(right_shoulder, right_wrist) > 0.45  # More extended
       
        wrist_forward = right_wrist[2] < right_elbow[2]
        
        left_hand_relaxed = left_wrist[2] > right_wrist[2] + 0.1
        
        wrist_at_punch_height = abs(right_wrist[1] - right_shoulder[1]) < 0.2

        is_fast = False
        if self.previous_wrist_positions["right"]:
            movement = abs(right_wrist[2] - self.previous_wrist_positions["right"][2])
            is_fast = movement > 0.03
        
        self.previous_wrist_positions["right"] = right_wrist # type: ignore
        
        return (arm_extension and elbow_extended and wrist_forward and 
                left_hand_relaxed and wrist_at_punch_height and is_fast)
    
    def detect_kick_left(self, landmarks):
        """Detect left kick - left knee raised"""
        left_hip = self.get_landmark_coords(landmarks, 23)
        left_knee = self.get_landmark_coords(landmarks, 25)
        left_ankle = self.get_landmark_coords(landmarks, 27)
        
        knee_raised = left_knee[1] < left_hip[1] - 0.2  # Stricter threshold
        
        ankle_raised = left_ankle[1] < left_hip[1]
        
        return knee_raised and ankle_raised
    
    def detect_kick_right(self, landmarks):
        """Detect right kick"""
        right_hip = self.get_landmark_coords(landmarks, 24)
        right_knee = self.get_landmark_coords(landmarks, 26)
        right_ankle = self.get_landmark_coords(landmarks, 28)
        

        knee_raised = right_knee[1] < right_hip[1] - 0.2
        
        ankle_raised = right_ankle[1] < right_hip[1]
        
        return knee_raised and ankle_raised
    
    def detect_block(self, landmarks):
        """Detect block - both hands raised near face"""
        nose = self.get_landmark_coords(landmarks, 0)
        left_wrist = self.get_landmark_coords(landmarks, 15)
        right_wrist = self.get_landmark_coords(landmarks, 16)
        left_shoulder = self.get_landmark_coords(landmarks, 11)
        right_shoulder = self.get_landmark_coords(landmarks, 12)
        
        left_hand_up = left_wrist[1] < left_shoulder[1] - 0.1
        right_hand_up = right_wrist[1] < right_shoulder[1] - 0.1
        

        left_hand_near_face = abs(left_wrist[2] - nose[2]) < 0.3
        right_hand_near_face = abs(right_wrist[2] - nose[2]) < 0.3
        
        return left_hand_up and right_hand_up and left_hand_near_face and right_hand_near_face
    
    def detect_crouch(self, landmarks):
        """Detect crouch - hips lowered"""
        left_shoulder = self.get_landmark_coords(landmarks, 11)
        right_shoulder = self.get_landmark_coords(landmarks, 12)
        left_hip = self.get_landmark_coords(landmarks, 23)
        right_hip = self.get_landmark_coords(landmarks, 24)
        left_knee = self.get_landmark_coords(landmarks, 25)
        
        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        avg_hip_y = (left_hip[1] + right_hip[1]) / 2
        

        torso_compressed = abs(avg_shoulder_y - avg_hip_y) < 0.25
        

        knee_bent = abs(left_knee[1] - left_hip[1]) < 0.3
        
        return torso_compressed and knee_bent
    
    def detect_jump(self, landmarks):
        """Detect jump - both arms raised high above head"""
        nose = self.get_landmark_coords(landmarks, 0)
        left_shoulder = self.get_landmark_coords(landmarks, 11)
        right_shoulder = self.get_landmark_coords(landmarks, 12)
        left_wrist = self.get_landmark_coords(landmarks, 15)
        right_wrist = self.get_landmark_coords(landmarks, 16)
        
        # Both hands significantly above shoulders
        left_arm_up = left_wrist[1] < left_shoulder[1] - 0.25
        right_arm_up = right_wrist[1] < right_shoulder[1] - 0.25
        
        # Both hands above nose level
        left_above_nose = left_wrist[1] < nose[1]
        right_above_nose = right_wrist[1] < nose[1]
        
        return left_arm_up and right_arm_up and left_above_nose and right_above_nose
    
    def detect_move_left(self, landmarks):
        """Detect body lean to the left"""
        nose = self.get_landmark_coords(landmarks, 0)
        left_hip = self.get_landmark_coords(landmarks, 23)
        left_shoulder = self.get_landmark_coords(landmarks, 11)
        
        nose_left = nose[0] < left_hip[0] - 0.15
        shoulder_left = left_shoulder[0] < left_hip[0] - 0.1
        
        return nose_left and shoulder_left
    
    def detect_move_right(self, landmarks):
        """Detect body lean to the right"""
        nose = self.get_landmark_coords(landmarks, 0)
        right_hip = self.get_landmark_coords(landmarks, 24)
        right_shoulder = self.get_landmark_coords(landmarks, 12)
        
        # Nose and shoulder significantly right of hip
        nose_right = nose[0] > right_hip[0] + 0.15
        shoulder_right = right_shoulder[0] > right_hip[0] + 0.1
        
        return nose_right and shoulder_right
    
    def recognize_gesture(self, landmarks):
        """Main function to recognize all gestures"""
        if not landmarks:
            return "none"
        
        # Priority order (check block and crouch before punches/kicks)
        if self.detect_block(landmarks):
            return "block"
        if self.detect_jump(landmarks):
            return "jump"
        if self.detect_crouch(landmarks):
            return "crouch"
        if self.detect_punch_left(landmarks):
            return "punch_left"
        if self.detect_punch_right(landmarks):
            return "punch_right"
        if self.detect_kick_left(landmarks):
            return "kick_left"
        if self.detect_kick_right(landmarks):
            return "kick_right"
        if self.detect_move_left(landmarks):
            return "move_left"
        if self.detect_move_right(landmarks):
            return "move_right"
        
        return "none"
    
def execute_game_action(gesture):
    """Map gestures to keyboard inputs for Shadow Fight Arena"""
    action_map = {
        "punch_left": "j",
        "punch_right": "k",
        "kick_left": "u",
        "kick_right": "i",
        "block": "l", 
        "jump": "w",
        "crouch": "s", 
        "move_left": "a",
        "move_right": "d",
    }
    
    if gesture in action_map:
        pyautogui.press(action_map[gesture])
        return action_map[gesture]
    return None
    

