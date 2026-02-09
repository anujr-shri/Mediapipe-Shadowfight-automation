import numpy as np
import pyautogui
class GestureRecognizer:
    def __init__(self):
        self.previous_gesture = None
        self.gesture_threshold = 0.15
        
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
        
        arm_extension = left_wrist[2] < left_shoulder[2] - 0.1
        elbow_extended = self.calculate_distance(left_shoulder, left_wrist) > 0.3
        
        return arm_extension and elbow_extended
    
    def detect_punch_right(self, landmarks):
        """Detect right punch"""
        right_shoulder = self.get_landmark_coords(landmarks, 12)
        right_elbow = self.get_landmark_coords(landmarks, 14)
        right_wrist = self.get_landmark_coords(landmarks, 16)
        
        arm_extension = right_wrist[2] < right_shoulder[2] - 0.1
        elbow_extended = self.calculate_distance(right_shoulder, right_wrist) > 0.3
        
        return arm_extension and elbow_extended
    
    def detect_kick_left(self, landmarks):
        """Detect left kick - left knee raised"""
        left_hip = self.get_landmark_coords(landmarks, 23)
        left_knee = self.get_landmark_coords(landmarks, 25)
        
        return left_knee[1] < left_hip[1] - 0.15
    
    def detect_kick_right(self, landmarks):
        """Detect right kick"""
        right_hip = self.get_landmark_coords(landmarks, 24)
        right_knee = self.get_landmark_coords(landmarks, 26)
        
        return right_knee[1] < right_hip[1] - 0.15
    
    def detect_block(self, landmarks):
        """Detect block - both hands raised near face"""
        nose = self.get_landmark_coords(landmarks, 0)
        left_wrist = self.get_landmark_coords(landmarks, 15)
        right_wrist = self.get_landmark_coords(landmarks, 16)
        
        left_hand_up = left_wrist[1] < nose[1]
        right_hand_up = right_wrist[1] < nose[1]
        
        return left_hand_up and right_hand_up
    
    def detect_crouch(self, landmarks):
        """Detect crouch - hips lowered"""
        left_shoulder = self.get_landmark_coords(landmarks, 11)
        left_hip = self.get_landmark_coords(landmarks, 23)
        left_knee = self.get_landmark_coords(landmarks, 25)
        
        torso_compressed = abs(left_shoulder[1] - left_hip[1]) < 0.3
        
        return torso_compressed
    
    def detect_jump(self, landmarks):
        """Detect jump - both arms raised high"""
        left_shoulder = self.get_landmark_coords(landmarks, 11)
        right_shoulder = self.get_landmark_coords(landmarks, 12)
        left_wrist = self.get_landmark_coords(landmarks, 15)
        right_wrist = self.get_landmark_coords(landmarks, 16)
        
        left_arm_up = left_wrist[1] < left_shoulder[1] - 0.2
        right_arm_up = right_wrist[1] < right_shoulder[1] - 0.2
        
        return left_arm_up and right_arm_up
    
    def detect_move_left(self, landmarks):
        """Detect body lean to the left"""
        nose = self.get_landmark_coords(landmarks, 0)
        left_hip = self.get_landmark_coords(landmarks, 23)
        
        return nose[0] < left_hip[0] - 0.1
    
    def detect_move_right(self, landmarks):
        """Detect body lean to the right"""
        nose = self.get_landmark_coords(landmarks, 0)
        right_hip = self.get_landmark_coords(landmarks, 24)
        
        return nose[0] > right_hip[0] + 0.1
    
    def recognize_gesture(self, landmarks):
        """Main function to recognize all gestures"""
        if not landmarks:
            return "none"

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
    

