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
import scipy.io
# from datasets.utils.helpers import GraphGenerator, get_indices
from tqdm import tqdm
import re


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
        return len(self.y)
    
    def generate_graphs(self):
        '''
        Generate a dataset wide tensor to store spatio temporal datapoints
        '''
        Xs = []
        ys = []
        self.subjects = []
        for idx in tqdm(range(len(self.labels_df))):
            folder_name = re.sub(r'_\d+$', '', self.labels_df["Filename"][idx])
            frames_folder = os.path.join(self.frames_path, folder_name)
            apex = self.labels_df["Apex"][idx]
            onset = self.labels_df["Onset"][idx]
            offset = self.labels_df["Offset"][idx]
            subject = self.labels_df["Subject"][idx]
            label = self.labels_df["au4"][idx] # TODO: take au as input

            # data = scipy.io.loadmat(frames_path)
            # tempImg = np.squeeze(data["tempImg"])


            frames = [onset, (onset+apex)//2, apex, (apex+offset)//2, offset]
            # print(frames)
            # for frame in frames:
            #     print(str(frame).zfill(4))
            if folder_name != "016_7": #single exception to zfill(4) rule 
                # /media/thatblueboy/Seagate/LOP/data/SAMMLong/frames/016_7/016_7_1275.jpg
                image_frames = [os.path.join(frames_folder, str(folder_name)+"_"+str(frame).zfill(4)+".jpg") for frame in frames]
            else:
                image_frames = [os.path.join(frames_folder, str(folder_name)+"_"+str(frame).zfill(5)+".jpg") for frame in frames]


            images = []
            for t, image_path in enumerate(sorted(image_frames)):
                # print(image_path)
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images.append(gray)

            x, num_errors = self.graph_generator.generate(images)
            if num_errors >1:
                print("skipping: graph could not be generated", frames_folder, num_errors)
                print(idx)
                continue

            Xs.append(x)
            ys.append(torch.tensor(label, dtype=torch.float32))
            self.subjects.append(subject)
            

        self.X = torch.stack(Xs)
        self.y = torch.stack(ys)

    def __getitem__(self, idx):
        return self.X[idx], self.edge_index, self.y[idx], self.subjects[idx]