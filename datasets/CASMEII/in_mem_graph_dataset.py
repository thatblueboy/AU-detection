import os

import cv2
import torch
import numpy as np
from tqdm import tqdm

from datasets.utils.base_in_mem_graph_dataset import BaseInMemoryGraphDataset

class CASMEIIAUGraphDataset(BaseInMemoryGraphDataset): 
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
            label = self.labels_df["au"+str(self.au)][idx]
            images_folder = os.path.join(self.frames_path, "sub" + str(subject).zfill(2),self.labels_df["Filename"][idx])

            if self.num_timesteps==5:
                frames = [onset, (onset+apex)//2, apex, (apex+offset)//2, offset]
            else:
                frames = np.floor(np.linspace(onset, offset, self.num_timesteps)).astype(int)
                
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