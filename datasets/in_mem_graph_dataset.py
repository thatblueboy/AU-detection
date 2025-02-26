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
from datasets.utils.helpers import GraphGenerator, get_indices
from tqdm import tqdm

class InMemoryGraphDataset(Dataset): 
    '''
    Makes stacked Graphs of 5 frames from onset to offset
    input: frames_path, depth_path, labels_path, landmarks(different for different AU), edges(different for different AU)
    '''
    def __init__(self, 
                 frames_path, 
                 depth_path, 
                 labels_csv_path, 
                 landmarks, 
                 edges, 
                 image_size=(400, 400),
                 device='cpu'):
        
        landmarks = sorted(landmarks)
        self.graph_generator = GraphGenerator(landmarks)
        self.frames_path = frames_path
        self.depth_path = depth_path
        self.labels_df = pd.read_csv(labels_csv_path)
        self.image_size = image_size
        self.edges = edges
        self.include_flips = True
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
        Xs = []
        ys = []
        self.subjects = []
        for idx in tqdm(range(len(self.labels_df))):
            images_folder = os.path.join(self.frames_path,self.labels_df["Subfolder Name"][idx])
            depth_folder = os.path.join(self.depth_path,self.labels_df["Subfolder Name"][idx])
            apex = self.labels_df["Apex"][idx]
            onset = self.labels_df["Onset"][idx]
            offset = self.labels_df["Offset"][idx]
            subject = self.labels_df["Subject"][idx]
            label = self.labels_df["label"][idx]

            imgs = get_indices(images_folder, onset, apex, offset)        

            image_frames = [os.path.join(images_folder,str(file)+".jpg") for file in imgs]
            depth_frames = [os.path.join(depth_folder,str(file)+".png") for file in imgs]

            gray_images =[]
            depth_images =[]

            for t, (image_path, depth_path) in enumerate(zip(sorted(image_frames), sorted(depth_frames))):
                image = cv2.imread(image_path)
                # if image.shape[:2] != self.image_size: #some images are not properly sized
                #     print("not properly sized", images_folder)
                #     image = cv2.resize(image, self.image_size)

                depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                # if depth_image.shape != self.image_size: #some images are not properly sized
                #     depth_image = cv2.resize(depth_image, self.image_size)
                # Convert the RGB image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_images.append(gray)
                depth_images.append(depth_image)
               
            x, num_errors = self.graph_generator.generate(gray_images, depth_images)

           
            Xs.append(x)
            ys.append(torch.tensor(label, dtype=torch.float32))
            self.subjects.append(subject)
            if num_errors >1:
                print(images_folder, num_errors)
                print(idx)
            if self.include_flips: #flip
                x, num_errors = self.graph_generator.generate([cv2.flip(img, 1) for img in gray_images],
                                                            [cv2.flip(img, 1) for img in depth_images])
               
                Xs.append(x)
                ys.append(torch.tensor(label, dtype=torch.float32))
                self.subjects.append(subject)
            if num_errors >1:
                print("flipped1", images_folder, num_errors)
                print(idx)

        self.X = torch.stack(Xs)
        self.y = torch.stack(ys)

    def __getitem__(self, idx):
        return self.X[idx], self.edge_index, self.y[idx], self.subjects[idx]

    # def get(self, idx):
        
    #     images_folder = os.path.join(self.frames_path,self.labels_df["Subfolder Name"][idx])
    #     depth_folder = os.path.join(self.depth_path,self.labels_df["Subfolder Name"][idx])
    #     apex = self.labels_df["Apex"][idx]
    #     onset = self.labels_df["Onset"][idx]
    #     offset = self.labels_df["Offset"][idx]
    #     subject = self.labels_df["Subject"][idx]
    #     label = self.labels_df["label"][idx]

    #     imgs = get_indices(images_folder, onset, apex, offset)        

    #     image_frames = [os.path.join(images_folder,str(file)+".jpg") for file in imgs]
    #     depth_frames = [os.path.join(depth_folder,str(file)+".png") for file in imgs]

        
    #     gray_images =[]
    #     depth_images =[]

    #     for t, (image_path, depth_path) in enumerate(zip(sorted(image_frames), sorted(depth_frames))):
    #            image = cv2.imread(image_path)
    #            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    #            # Convert the RGB image to grayscale
    #            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #            gray_images.append(gray)
    #            depth_images.append(depth_image)
               
    #     X, num_errors = self.graph_generator.generate(gray_images, depth_images)
    #     label = torch.tensor(label, dtype=torch.float32)   # Convert label to int64 (long)

    #     return X, label, subject