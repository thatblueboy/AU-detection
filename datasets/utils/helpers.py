import dlib
import cv2
import torch
import os

def get_indices(folder_path, onset, apex, offset):
    '''
    Return 5 set of image paths from folder_path starting with onset, apex in middle
    and offset at end
    '''

    n_values = sorted(int(f.split(".")[0]) for f in os.listdir(folder_path) if f.endswith(".jpg"))
    def middle_frame(n1, n2):
        # Ensure n1 and n2 exist in the list
        if n1 not in n_values or n2 not in n_values:
            raise ValueError("One or both given n values do not exist in the folder.")

        # Get the sublist between n1 and n2 (inclusive)
        idx1, idx2 = n_values.index(n1), n_values.index(n2)
        middle_idx = (idx1 + idx2) // 2
        return n_values[middle_idx]
    
    return [onset, middle_frame(onset, apex), apex, middle_frame(apex, offset), offset]

class GraphGenerator():
    def __init__(self, landmarks, normalizing_factor=400, central_landmark=28):
        self.landmarks = landmarks
        self.central_landmark = central_landmark
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("/media/thatblueboy/Seagate/LOP/datasets/utils/shape_predictor_model")  # Path to landmarks model
        self.num_nodes = len(landmarks)
        self.normalizing_factor = normalizing_factor

    def generate(self, images):
        '''
        converts lists of frames into stacked landmarks
        with (x, y, depth) features
        '''

        X = torch.zeros([len(images), self.num_nodes, 2])
        err = 0
        for t, image in enumerate(images):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                faces = self.face_detector(image)
                if len(faces) == 0:
                    err += 1
                    print(f"No face detected. Skipping...", t)
                    continue

                # Use the first detected face (assuming one face per frame)
                face = faces[0]
                landmarks = self.shape_predictor(image, face)

                # Find x_c, y_c
                x_c, y_c = landmarks.part(self.central_landmark-1).x, landmarks.part(self.central_landmark-1).y

                # Find feature and store normalized
                for i, landmark in enumerate(self.landmarks):
                    x, y = landmarks.part(landmark - 1).x, landmarks.part(landmark - 1).y
                   
                    x_norm =x
                    y_norm = y
                    # Normalize relative to center node
                    X[t, i] = torch.tensor([
                        (x_norm - x_c) / self.normalizing_factor, #400
                        (y_norm - y_c) / self.normalizing_factor, #400 
                    ], dtype=X.dtype, device=X.device)

        if err>1:
                print("More than 1 timestep has no detected face!:", err)

        return X, err