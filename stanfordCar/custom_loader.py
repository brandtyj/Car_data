from torch.utils.data import Dataset
import os
from PIL import Image

class CustomLoader(Dataset):
    def __init__(self, list_label, root_dir, transformation, transform = False):
        self.annotations = list_label
        self.transformation = transformation
        self.transform = transform
        self.root_dir = root_dir
        
        
    def __len__(self):
        return len(os.listdir(self.root_dir)) # # of images 
    
    def __getitem__(self, index):
        '''Return specific item of index'''
        items = os.listdir(self.root_dir)

        row = self.annotations.loc[self.annotations[0] == items[index]]
        
        img_id = row[0].item()
        img_label = row[1].item()
        
        img_path = os.path.join(str(self.root_dir), str(img_id)) 
        
        image = Image.open(img_path).convert('RGB')
        
        # transform 
        if self.transform:
            image = self.transformation(image)
        return image, img_label
    


