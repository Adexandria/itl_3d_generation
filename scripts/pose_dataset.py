import torch
from torch.utils.data import Dataset
import pickle
import json
from transformers import BertTokenizer, BertModel   
from pytorch3d import transforms 
import os  

torch.serialization.register_package(0, lambda x: x.device.type, lambda x, _: x.cpu())

class PoseDataset(Dataset):
    def __init__(self, gloss_file, device='cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.path = os.path.dirname(gloss_file)
        with open(gloss_file, 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)
    
    def get_glosses(self,gloss):
        with torch.no_grad():
            tokens = self.tokenizer(gloss, return_tensors='pt').to(self.device)
            gloss_embedding = self.bert_model(**tokens).last_hidden_state[:, 0, :] 
        return gloss_embedding[0]

    def get_dataset(self, gloss):
        item = self.find_data(self.data, gloss)
        if item is None:
            return None
        
        # Extract gloss and frame count from the item
        gloss = item['text']
        if gloss is None:
            return None
        
        # Get gloss embedding
        gloss_embedding = self.get_glosses(gloss).to(self.device)

        frame_length = item['frame_count']
        if frame_length is None:
            return None,None
        
        return gloss_embedding, frame_length

    def __getitem__(self, idx):
        gloss = self.data[idx]['gloss']
        pkl_path = self.data[idx]['keypoints']
        pkl_path = os.path.join(self.path, pkl_path)
        pkl_path = os.path.normpath(pkl_path)  # Normalize the path
        
        # Get gloss embedding
        gloss_embedding = self.get_glosses(gloss)
        
        # Load SMPL-X keypoints
        try:
            with open(pkl_path, 'rb') as f:
                keypoints = pickle.load(f)  
        except FileNotFoundError:
            return None  # Handle missing files gracefully
        
        pose_vector = self.__transform_keypoints__(keypoints)
        
        gloss_embedding = torch.as_tensor(gloss_embedding).to(self.device)
        pose_vector = torch.as_tensor(pose_vector).to(self.device)
        
        return gloss_embedding,pose_vector

    def __transform_keypoints__(self, keypoints):
        betas = torch.stack(keypoints['betas'].get('body'))     
        betas = betas.squeeze(1)[:,:10]

        expression = torch.stack(keypoints['expression'].get('body'))
        expression = expression.squeeze(1)[:,:10]

        # Extract rotation matrices and convert to angle-axis representation
        rot_mats = torch.stack(keypoints['rot_mats'].get('body'))
        rot_mats = rot_mats.squeeze(1) 
                
        B = rot_mats.shape[0]
        camera = torch.stack(keypoints['camera']).squeeze(1)  # shape (B,3)
        if camera.ndim == 1:
            camera_ts = [camera] * B
        else:
            camera_ts = camera  # assume shape (B,3)   

        pose  = transforms.matrix_to_axis_angle(rot_mats)  # Convert rotation matrices to angle-axis representation
        pose = pose.reshape(B, -1)  # Flatten the pose to a single vector
        global_orient = pose[:, :3].to(self.device)  # Global orientation
        body_pose = pose[:, 3:66].to(self.device)  # Body pose
        jaw_pose = pose[:, 66:69].to(self.device)   # Jaw pose
        leye_pose= pose[:, 69:72].to(self.device)   # Left eye pose
        reye_pose = pose[:, 72:75].to(self.device)   # Right eye pose
        left_hand_pose = pose[:, 75:120].to(self.device)   # Left hand pose
        right_hand_pose = pose[:, 120:].to(self.device)   # Right hand pose

        betas            = betas.to(self.device) # [B,10]
        expression       = expression.to(self.device) # [B,10]
            

        pose_array = [
            betas,
            expression,
            global_orient,
            body_pose,
            jaw_pose,
            leye_pose,
            reye_pose,
            left_hand_pose,
            right_hand_pose,
            camera_ts,
        ]
    
        pose = torch.cat(pose_array, dim=1)  # Concatenate all parts into a single vector
        return pose

    
    def find_data(self,isolated_list, target_name):
        for item in isolated_list:
            if item['text'] == target_name:
                return item
        return None