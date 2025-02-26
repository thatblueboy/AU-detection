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
    def __init__(self, landmarks, central_landmark=28):
        self.landmarks = landmarks
        self.central_landmark = central_landmark
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("/media/thatblueboy/Seagate/LOP/datasets/utils/shape_predictor_model")  # Path to landmarks model
        self.num_nodes = len(landmarks)

    def generate(self, gray_images, depth_images):
        '''
        converts lists of frames and depth frames into stacked landmarks
        with (x, y, depth) features
        '''

        X = torch.zeros([len(gray_images), self.num_nodes, 3])
        err = 0
        for t, (gray_image, depth_image) in enumerate(zip(gray_images, depth_images)):
                faces = self.face_detector(gray_image)
                if len(faces) == 0:
                    err += 1
                    print(f"No face detected. Skipping...", t)
                    continue

                # Use the first detected face (assuming one face per frame)
                face = faces[0]
                landmarks = self.shape_predictor(gray_image, face)

                # Find z_avg amongst non 0 values
                z_values = []  # Store nonzero z-values
                for i, landmark in enumerate(self.landmarks):
                    x, y = landmarks.part(landmark - 1).x, landmarks.part(landmark - 1).y
                    z = depth_image[y, x]
                    if z > 200:  # Collect nonzero z-values
                        z_values.append(z)
                z_avg = sum(z_values) / len(z_values)  # Compute average


                # Find x_c, y_c, z_c
                x_c, y_c = landmarks.part(self.central_landmark-1).x, landmarks.part(self.central_landmark-1).y
                z_c = depth_image[y_c, x_c]
                z_c = z_c if z_c > 200 else z_avg

                # Find feature and store normalized
                for i, landmark in enumerate(self.landmarks):
                    x, y = landmarks.part(landmark - 1).x, landmarks.part(landmark - 1).y
                    z = depth_image[y, x]
                    z = z if z > 200 else z_avg  # Replace 0 with average
                    if z<200:
                        print("err")
                    # Normalize to [0,1] range
                    # x_norm = x / 400
                    # y_norm = y / 400
                    # z_norm = (z - 400) / (1300 - 400)
                    x_norm =x
                    y_norm = y
                    z_norm = z
                    # Normalize relative to center node
                    X[t, i] = torch.tensor([
                        x_norm - x_c / 400, 
                        y_norm - y_c / 400, 
                        z_norm - (z_c - 400) / (1300 - 400)
                    ], dtype=X.dtype, device=X.device)


        if err>1:
                print("More than 1 timestep has no detected face!:", err)

        return X, err

