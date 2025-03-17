from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import os
import pandas as pd
import dlib
from torchvision import transforms
import cv2
from datasets.utils.helpers import get_indices
from tqdm import tqdm


class GraphGenerator():
    def __init__(self, landmarks, central_landmark=28):
        self.landmarks = landmarks
        self.central_landmark = central_landmark
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("/media/thatblueboy/Seagate/LOP/datasets/utils/shape_predictor_model")  # Path to landmarks model
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
                        (x_norm - x_c) / 400, 
                        (y_norm - y_c) / 400, 
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
                 frames_path, 
                 labels_csv_path, 
                 landmarks, 
                 edges, 
                 image_size=(400, 400),
                 device='cpu'):
        
        landmarks = sorted(landmarks)
        self.graph_generator = GraphGenerator(landmarks)
        self.frames_path = frames_path
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
        return len(self.labels_df)
    
    def generate_graphs(self):
        '''
        Generate a dataset wide tensor to store spatio temporal datapoints
        '''
        # /media/thatblueboy/Seagate/LOP/data/CASMEII/Cropped/sub01/EP02_01f
        # /media/thatblueboy/Seagate/LOP/data/CASMEII/Cropped/sub01/EP02_01f/reg_img48.jpg
        Xs = []
        ys = []
        self.subjects = []
        for idx in tqdm(range(len(self.labels_df))):
            apex = int(self.labels_df["ApexFrame"][idx])
            onset = int(self.labels_df["OnsetFrame"][idx])
            offset = int(self.labels_df["OffsetFrame"][idx])
            subject = self.labels_df["Subject"][idx]
            label = self.labels_df["au4"][idx] #TODO: take as input
            images_folder = os.path.join(self.frames_path, "sub" + str(subject).zfill(2),self.labels_df["Filename"][idx])

            frames = [onset, (onset+apex)//2, apex, (apex+offset)//2, offset]
            
            image_frames = [os.path.join(images_folder, "reg_img"+str(frame)+".jpg") for frame in frames]

            gray_images =[]
            for t, image_path in enumerate(image_frames):
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_images.append(gray)
               
            x, num_errors = self.graph_generator.generate(gray_images)
           
            Xs.append(x)
            ys.append(torch.tensor(label, dtype=torch.float32))
            self.subjects.append(subject)
            if num_errors >1:
                print(images_folder, num_errors)
                print(idx)
          
        self.X = torch.stack(Xs)
        self.y = torch.stack(ys)

    def __getitem__(self, idx):
        return self.X[idx], self.edge_index, self.y[idx], self.subjects[idx]
