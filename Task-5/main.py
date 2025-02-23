import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import mediapipe as mp

# Load FaceNet model (pre-trained on VGGFace2)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Set up MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def get_alignment_matrix(left_eye, right_eye, desired_left=(40, 60), desired_right=(120, 60)):
    """
    Compute the affine transformation matrix to align the face based on eye positions.
    
    Args:
        left_eye (tuple): (x, y) coordinates of the left eye in pixels.
        right_eye (tuple): (x, y) coordinates of the right eye in pixels.
        desired_left (tuple): Desired (x, y) position of the left eye in the aligned image.
        desired_right (tuple): Desired (x, y) position of the right eye in the aligned image.
    
    Returns:
        numpy.ndarray: 2x3 affine transformation matrix.
    """
    left_eye = np.array(left_eye)
    right_eye = np.array(right_eye)
    v = right_eye - left_eye
    d = np.linalg.norm(v)
    s = (desired_right[0] - desired_left[0]) / d  # Scale factor (desired distance is 80 pixels)
    theta = np.arctan2(v[1], v[0])
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    A = s * np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
    b = np.array(desired_left) - A @ left_eye
    M = np.hstack((A, b.reshape(2,1)))
    return M

def get_aligned_face(image, detection, image_size=160):
    """
    Detect and align a face in the image using MediaPipe keypoints.
    
    Args:
        image (numpy.ndarray): RGB image.
        detection (mediapipe.framework.formats.detection.Detection): Face detection result.
        image_size (int): Size of the output aligned face image (default is 160x160).
    
    Returns:
        numpy.ndarray: Aligned face crop of shape (image_size, image_size, 3).
    """
    h, w = image.shape[:2]
    # Get keypoints (relative coordinates)
    keypoints = detection.location_data.relative_keypoints
    left_eye = (keypoints[0].x * w, keypoints[0].y * h)
    right_eye = (keypoints[1].x * w, keypoints[1].y * h)
    # Compute alignment matrix
    M = get_alignment_matrix(left_eye, right_eye)
    # Warp and crop
    aligned_face = cv2.warpAffine(image, M, (image_size, image_size), flags=cv2.INTER_LINEAR)
    return aligned_face

def compute_embedding(aligned_face):
    """
    Compute the FaceNet embedding for an aligned face image.
    
    Args:
        aligned_face (numpy.ndarray): Aligned face image of shape (160, 160, 3) in RGB.
    
    Returns:
        numpy.ndarray: 512-dimensional embedding vector.
    """
    # Convert to float32 and normalize to [-1, 1]
    aligned_face = aligned_face.astype(np.float32) / 255.0
    aligned_face = aligned_face * 2 - 1
    # Convert to PyTorch tensor (shape: [1, 3, 160, 160])
    tensor = torch.from_numpy(aligned_face).permute(2, 0, 1).unsqueeze(0)
    # Compute embedding
    with torch.no_grad():
        embedding = model(tensor).numpy()[0]
    return embedding

# Define known faces (replace with your own images and names)
known_faces = [
    ("Jeff Bezos", "Task-5/person1.jpeg"),
    ("Elon Musk", "Task-5/person2.jpeg"),
    # Add more as needed
]

# Compute embeddings for known faces
known_embeddings = []
known_names = []

for name, image_path in known_faces:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)
    if results.detections:
        detection = results.detections[0]  # Assume one face per image
        aligned_face = get_aligned_face(rgb_image, detection)
        embedding = compute_embedding(aligned_face)
        known_embeddings.append(embedding)
        known_names.append(name)
    else:
        print(f"No face detected in {image_path}")

# Start webcam for real-time recognition
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    if results.detections:
        for detection in results.detections:
            aligned_face = get_aligned_face(rgb_frame, detection)
            embedding = compute_embedding(aligned_face)
            # Compare to known embeddings
            distances = [np.linalg.norm(embedding - known_emb) for known_emb in known_embeddings]
            min_distance = min(distances) if distances else float('inf')
            if min_distance < 0.6:  # Recognition threshold (adjust as needed)
                name = known_names[np.argmin(distances)]
            else:
                name = "Unknown"
            # Draw bounding box and name on the original frame
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Clean up MediaPipe resources
face_detection.close()