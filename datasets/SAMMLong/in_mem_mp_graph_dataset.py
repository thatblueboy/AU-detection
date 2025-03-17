from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import os
import pandas as pd
import mediapipe as mp
from torchvision import transforms
import cv2
import scipy.io
# from datasets.utils.helpers import GraphGenerator, get_indices
from tqdm import tqdm


class GraphGenerator():
    def __init__(self, landmarks, central_landmark=28):
        self.landmarks = landmarks
        self.central_landmark = central_landmark
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

        # self.face_detector = dlib.get_frontal_face_detector()
        # self.shape_predictor = dlib.shape_predictor("/media/thatblueboy/Seagate/LOP/datasets/utils/shape_predictor_model")  # Path to landmarks model
        self.num_nodes = len(landmarks)

    def generate(self, images):
        '''
        converts lists of frames into stacked landmarks
        with (x, y, depth) features
        '''

        X = torch.zeros([len(images), self.num_nodes, 2])
        err = 0
        for t, image in enumerate(images):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                result = self.face_mesh.process(image)

                if not result.multi_face_landmarks:
                    err += 1
                    print(f"No face detected. Skipping...", t)
                    continue
                
                #first face
                landmarks = result.multi_face_landmarks[0].landmark

                # Find x_c, y_c
                x_c, y_c = landmarks[self.central_landmark].x, landmarks[self.central_landmark].y

                # Find feature and store normalized
                for i, landmark in enumerate(self.landmarks):
                    x, y = landmarks[landmark].x, landmarks[landmark].y
                   
                    # Normalize relative to center node
                    X[t, i] = torch.tensor([
                        x - x_c, 
                        y - y_c, 
                    ], dtype=X.dtype, device=X.device)

        if err>1:
                print("More than 1 timestep has no detected face!:", err)

        return X, err



class InMemoryGraphDataset(Dataset): 
    '''
    Makes stacked Graphs of 5 frames from onset to offset
    input: frames_path, depth_path, labels_path, landmarks(different for different AU), edges(different for different AU)
    '''
    def __init__(self, 
                 mat_path, 
                 labels_csv_path, 
                 centre,
                 landmarks, 
                 edges, 
                 image_size=(400, 400),
                 device='cpu'):
        
        landmarks = sorted(landmarks)
        self.graph_generator = GraphGenerator(landmarks, central_landmark=centre)
        self.mat_path = mat_path
        self.labels_df = pd.read_csv(labels_csv_path)
        self.image_size = image_size
        self.edges = edges
        self.device = device

        edge_list = []
        node_mapping = {node: i for i, node in enumerate(landmarks)}
        for src, dst in edges:
            edge_list.append((node_mapping[src], node_mapping[dst]))  # Forward edge
            edge_list.append((node_mapping[dst], node_mapping[src]))  # Reverse edge (undirected)

        self.edge_index = torch.tensor(edge_list, dtype=torch.long).T
 
        self.generate_graphs()
        
        self.X = self.X.to(self.device)
        self.y = self.y.to(self.device)

    def __len__(self):
        return len(self.y)
    
    def generate_graphs(self):
        '''
        Generate a dataset wide tensor to store spatio temporal datapoints
        '''
        Xs = []
        ys = []
        self.subjects = []
        for idx in tqdm(range(len(self.labels_df))):
            mat = os.path.join(self.mat_path, self.labels_df["Filename"][idx]+".mat")
            apex = self.labels_df["Apex Frame"][idx]
            onset = self.labels_df["Onset Frame"][idx]
            offset = self.labels_df["Offset Frame"][idx]
            subject = self.labels_df["Subject"][idx]
            label = self.labels_df["au4"][idx] # TODO: take au as input

            data = scipy.io.loadmat(mat)
            tempImg = np.squeeze(data["tempImg"])

            onset, apex, offset = onset-onset, apex-onset, offset-onset 

            indices = [onset, (onset+apex)//2, apex, (apex+offset)//2, offset]
            images = [tempImg[:, :, i] for i in indices]

            x, num_errors = self.graph_generator.generate(images)
            if num_errors >1:
                print("skipping: graph could not be generated", mat, num_errors)
                print(idx)
                continue

            Xs.append(x)
            ys.append(torch.tensor(label, dtype=torch.float32))
            self.subjects.append(subject)
            
        
        self.X = torch.stack(Xs)
        self.y = torch.stack(ys)

    def __getitem__(self, idx):
        return self.X[idx], self.edge_index, self.y[idx], self.subjects[idx]