import os
import re

import cv2
import torch
import numpy as np
from tqdm import tqdm

from datasets.utils.base_in_mem_graph_dataset import BaseInMemoryGraphDataset

class SAMMLongAUGraphDataset(BaseInMemoryGraphDataset): 
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
            label=[]
            for au in self.aus:
                label.append(self.labels_df["au" + str(au)][idx])

            if self.num_timesteps==5:
                frames = [onset, (onset+apex)//2, apex, (apex+offset)//2, offset]
            else:
                frames = np.floor(np.linspace(onset, offset, self.num_timesteps)).astype(int)
                
            if folder_name != "016_7": #single exception to zfill(4) rule 
                # /media/thatblueboy/Seagate/LOP/data/SAMMLong/frames/016_7/016_7_1275.jpg
                image_frames = [os.path.join(frames_folder, str(folder_name)+"_"+str(frame).zfill(4)+".jpg") for frame in frames]
            else:
                image_frames = [os.path.join(frames_folder, str(folder_name)+"_"+str(frame).zfill(5)+".jpg") for frame in frames]

            images = []
            for t, image_path in enumerate(sorted(image_frames)):
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images.append(gray)
            
            x, num_errors = self.graph_generator.generate(images)
            if num_errors >1:
                print("skipping: graph could not be generated", frames_folder, num_errors)
                print(idx)
            else:
                Xs.append(x)
                ys.append(torch.tensor(label, dtype=torch.float32))
                self.subjects.append(subject)            

            if self.include_flipped: #flip
                x, num_errors = self.graph_generator.generate([cv2.flip(img, 1) for img in images])
                if num_errors >1:
                    print("skipping flipped image: graph could not be generated", frames_folder, num_errors)
                    print(idx)
                else:
                    Xs.append(x)
                    ys.append(torch.tensor(label, dtype=torch.float32))
                    self.subjects.append(subject)        

        self.X = torch.stack(Xs)
        self.all_au_labels = torch.stack(ys)