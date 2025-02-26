from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import os
import pandas as pd

from torchvision import transforms

def preprocess_and_combine(frame, depth_tensor, transform, image_size=(227, 227)):
    '''
    Make sure image and depth both have same size as input, 
    remove outliers in depth image.
    '''
    depth_tensor = transforms.ToTensor()(depth_tensor)
    frame_tensor = transforms.ToTensor()(frame)

    # Concatenate RGB and depth tensors
    combined_tensor = torch.cat((frame_tensor, depth_tensor), dim=0)  # Shape: (4, H, W)

    # Apply combined transforms (resize and normalize)
   
    combined_tensor = transform(combined_tensor)

    return combined_tensor
    
class ApexDataset(Dataset):
    def __init__(self, frames_path, depth_path, labels_csv_path, transform=None, image_size=(227, 227)):
        self.frames_path = frames_path
        self.depth_path = depth_path
        self.labels_df = pd.read_csv(labels_csv_path)
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        
        # depth_file_name = f""
        apex_depth_tensor_path = os.path.join(self.depth_path,self.labels_df["Subfolder Name"][idx],f"{int(self.labels_df['Apex'][idx])}.png")
        depth_tensor = Image.open(apex_depth_tensor_path).convert('L')

        apex_image_path = os.path.join(self.frames_path,self.labels_df["Subfolder Name"][idx],f"{int(self.labels_df['Apex'][idx])}.jpg")
        image = Image.open(apex_image_path).convert('RGB')

        four_class_label = self.labels_df["encoded_four_class_emotion"][idx]
        seven_class_label = self.labels_df["encoded_seven_class_emotion"][idx]
        
        if self.transform:
            combined_tensor = preprocess_and_combine(image, depth_tensor, self.transform, self.image_size)
        else:
            default_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                # transforms.ToTensor(), # converts to [0, 1] range too
            ])
            combined_tensor = preprocess_and_combine(image, depth_tensor, default_transform, self.image_size)

        return combined_tensor, four_class_label
