from datasets.CASME3.graph_dataset import GraphDataset

class AU4GraphDataset(GraphDataset): #TODO use config file
    def __init__(self):
        images_folder = "/media/thatblueboy/Seagate/LOP/data/raw/part_A/data/Part_A_ME_clip/frame"
        depth_folder = "/media/thatblueboy/Seagate/LOP/data/raw/part_A/data/Part_A_ME_clip/depth"
        labels_csv = "/media/thatblueboy/Seagate/LOP/data/processed/cleaned_au4.csv"
        ROI_index  = list(range(18, 29))+list(range(37, 49))
        edges = [(18, 37), (19, 38), (20, 38), (21, 39), (22, 40), 
                      (22, 28), (18, 19), (19, 20), (20, 21), (21, 22),
                      (28, 23), (23, 43), (24, 44), (25, 45), (26, 45), (27, 46),
                      (23, 24), (24, 25), (25, 26), (26, 27),
                      (46, 45), (45, 44), (44, 43), (43, 48), (48, 47), (47, 46), (43, 46),
                      (37, 38), (38, 39), (39, 40), (40, 41), (40, 37), (41, 42), (42, 37),
                      (40, 28), (28, 43), (22, 23)]

        super().__init__(images_folder, depth_folder, labels_csv, ROI_index, edges, (400, 400))