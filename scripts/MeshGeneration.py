import torch
import os
import cv2
import ffmpeg
import smplx
import numpy as np
from rnn import RNN
from tqdm import tqdm
from pose_dataset import PoseDataset
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils.renderer import Renderer

DARK_BLUE = (0.1, 0.2, 0.4)


class MeshGenerator:
    def __init__(self, gloss, checkpoint, gloss_file ,out_folder='output', save_mesh=False, device='cpu'):
        self.device = device
        self.checkpoint = checkpoint
        self.out_folder = out_folder
        self.save_mesh = save_mesh
        self.text = gloss
        self.gloss_file = gloss_file

    def run(self):
        print("Starting mesh generation...")
       
        # create smplx model
        smplx_model = self.create_smplx_model()

        # create ITL3D model
        itl3d_model = self.create_ITL3D_model(self.checkpoint, self.device)
        if itl3d_model is None:
            raise ValueError(f"Model not found for checkpoint: {self.checkpoint}")
       
       # Download and load checkpoints for Hamer
        download_models(CACHE_DIR_HAMER)
        model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        
        print(f"Loaded model from {self.checkpoint}")
        smplx_model = smplx_model.to(self.device)
        model = model.to(self.device)
        itl3d_model = itl3d_model.to(self.device)
        model.eval()
        itl3d_model.eval()

        renderer = Renderer(model_cfg, faces= smplx_model.faces)

        pose_dataset = PoseDataset(gloss_file=self.gloss_file, device=self.device)

        gloss_embedding,frame_length =  pose_dataset.get_glosses(self.text)

        output = itl3d_model(gloss_embedding,frame_length)

        transformed_output = self.transform_pose(output)
        
        output = smplx_model(betas=transformed_output['betas'], expression=transformed_output['expression'],
                          global_orient = transformed_output['go_aa'],
                          body_pose = transformed_output['bp_aa'],
                          jaw_pose = transformed_output['jaw_pose'],
                          left_hand_pose = transformed_output['left_hand_pose'],
                          right_hand_pose=transformed_output['right_hand_pose'],
                          leye_pose = transformed_output['left_eye_pose'],
                          reye_pose = transformed_output['right_eye_pose'],
                   return_verts=True)
        
        vertices = output.vertices.detach().cpu().numpy()

        out_path = os.path.join(self.out_folder, self.text)
        
        os.makedirs(out_path, exist_ok=True)

        camera = transformed_output['camera_ts'].detach().cpu().numpy() # Ensure shape is (B, 3)

        print("camera shape", camera.shape)

        B = vertices[0]  # Number of samples

        print(f'Number of samples: {B}')

        if camera.ndim == 1:
            camera_ts = camera.repeat(B, 1)  # Repeat for each sample
            print("camera_ts:", camera_ts[:3])
        else:
            camera_ts = camera  # assume shape (B,3)


        for i in tqdm(range(B),desc="Generating meshes and images"):
            verts_i = vertices[i]  # (V,3)
            print(f'vertices {i}', verts_i.shape)

            cam_i = camera_ts[i]
            print(f'before camera {i} shape:', cam_i.shape)
            cam_i = cam_i.reshape(3)

            print(f"camera {i}", cam_i.min(), cam_i.max())
            print("vertices:", verts_i.shape, verts_i.min(), verts_i.max())

            cam_i[2] += 3.5  # Move back for full body
            cam_i[1] += 0.3

            image = torch.ones(3, 1024, 1024)

            regression_img = renderer(vertices=verts_i,
                                      camera_translation=cam_i,
                                      image=image,
                                      mesh_base_color=DARK_BLUE,
                                      scene_bg_color=(1.0, 1.0, 1.0))

            print(f'Regression image min/max: {regression_img.min()}, {regression_img.max()}')

            image_uint8 = (regression_img * 255).clip(0, 255).astype('uint8')
            image_output_path = os.path.join(out_path, 'images')
            os.makedirs(image_output_path, exist_ok=True)
            cv2.imwrite(os.path.join(image_output_path, f'{self.text}-{i}.png'), image_uint8[:, :, ::-1])

            if self.save_mesh:
                camera_translation = cam_i.copy()
                tmesh = renderer.vertices_to_trimesh(verts_i, camera_translation, DARK_BLUE)
                tmesh_path = os.path.join(out_path, 'meshes')
                os.makedirs(tmesh_path, exist_ok=True)
                tmesh.export(os.path.join(tmesh_path, f'{self.text}-{i}.obj'))


        ffmpeg.input(os.path.join(image_output_path, f'{self.text}-%d.png'),
                     pattern_type='sequence', framerate=24, start_number=1).output(                
            os.path.join(out_path, f'{self.text}_video.mp4'),
            vcodec='libx264', r=24, pix_fmt='yuv420p').overwrite_output().run()


    def create_smplx_model(self):
        model = smplx.create(model_path = "/kaggle/input/smplx-model", model_type='smplx',
                         gender='female',use_pca=False,num_betas=10,num_expression_coeffs=10,
                         ext='npz')
        return model
    
    def create_ITL3D_model(self,checkpoint,device='cpu'):
        model = RNN( gloss_dim=768,
                     output_dim= 188,
                     hidden_dim=128)
        checkpoint = torch.load(checkpoint, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def transform_pose(self, reconstructed_pose):
        reconstructed_pose = reconstructed_pose[:0,:]
        betas            = reconstructed_pose[:, :10]
        expression       = reconstructed_pose[:, 10:20]
        go_aa            = reconstructed_pose[:, 20:23]
        bp_aa            = reconstructed_pose[:, 23:86]
        jaw_pose         = reconstructed_pose[:, 86:89]
        left_hand_pose   = reconstructed_pose[:, 89:134]
        right_hand_pose  = reconstructed_pose[:, 134:179]
        left_eye_pose    = reconstructed_pose[:, 179:182]
        right_eye_pose   = reconstructed_pose[:, 182:185]
        camera_ts        = reconstructed_pose[:, 185:188]
        

        return {
            'betas': betas,
            'expression': expression,
            'go_aa': go_aa,
            'bp_aa': bp_aa,
            'jaw_pose': jaw_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'left_eye_pose': left_eye_pose,
            'right_eye_pose': right_eye_pose,
            'camera_ts': camera_ts
        }
    