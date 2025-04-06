import os
import torch

import cv2
import scipy.io
import numpy as np
from tqdm import tqdm

from datasets.utils.base_in_mem_graph_dataset import BaseInMemoryGraphDataset

class SAMMAUGraphDataset(BaseInMemoryGraphDataset): 
    def generate_graphs(self):
        '''
        Generate a dataset wide tensor to store spatio temporal datapoints
        '''
        Xs = []
        ys = []
        self.subjects = []
        for idx in tqdm(range(len(self.labels_df))):
            mat = os.path.join(self.frames_path, self.labels_df["Filename"][idx]+".mat")
            apex = self.labels_df["Apex Frame"][idx]
            onset = self.labels_df["Onset Frame"][idx]
            offset = self.labels_df["Offset Frame"][idx]
            subject = self.labels_df["Subject"][idx]
            label = []
            for au in self.aus:
                label.append(self.labels_df[f"au{au}"][idx])

            data = scipy.io.loadmat(mat)
            tempImg = np.squeeze(data["tempImg"])

            onset, apex, offset = onset-onset, apex-onset, offset-onset 

            if self.num_timesteps==5:
                indices = [onset, (onset+apex)//2, apex, (apex+offset)//2, offset]
            else:
                indices = [onset, (onset+apex)//2, apex, (apex+offset)//2, offset]

            images = [tempImg[:, :, i] for i in indices]

            x, num_errors = self.graph_generator.generate(images)
            if num_errors >1:
                print("skipping: graph could not be generated", mat, num_errors)
                print(idx)
            else:
                Xs.append(x)
                ys.append(torch.tensor(label, dtype=torch.float32))
                self.subjects.append(subject)            

            if self.include_flipped: #flip
                x, num_errors = self.graph_generator.generate([cv2.flip(img, 1) for img in images])
                if num_errors >1:
                    print("skipping flipped image: graph could not be generated", mat, num_errors)
                    print(idx)
                else:
                    Xs.append(x)
                    ys.append(torch.tensor(label, dtype=torch.float32))
                    self.subjects.append(subject)

        self.X = torch.stack(Xs)
        self.all_au_labels = torch.stack(ys)

