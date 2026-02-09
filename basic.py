import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils, PoseLandmarkerResult, PoseLandmarker
from mediapipe.tasks.python.vision import drawing_styles, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import cv2




def draw_landmarks_on_image(detection_result, rgb_image):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image


def main():
  run_mode = RunningMode
  frame_count = 0
  base_option = python.BaseOptions(model_asset_path="pose_landmarker_full.task")

  option = PoseLandmarkerOptions(
    base_options=base_option,
    running_mode=run_mode.VIDEO,
    num_poses=6,
  )

  with PoseLandmarker.create_from_options(option) as detector:
    cap = cv2.VideoCapture(0)

    while True:
      step, img = cap.read()

      rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      timestamp_ms = int(frame_count * 1000 / 30) 
      mp_img = mp.Image(data=rgb_img, image_format=mp.ImageFormat.SRGB)
      frame_count += 1
      result = detector.detect_for_video(mp_img, timestamp_ms)
      annoted_img = draw_landmarks_on_image(result, mp_img.numpy_view())
      rgd_annoted_img = cv2.cvtColor(annoted_img, cv2.COLOR_RGB2BGR)
      cv2.imshow('pose Detection',rgd_annoted_img)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()









