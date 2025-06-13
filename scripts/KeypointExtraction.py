import torch
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict
from pytorch3d import transforms
from detectron2 import model_zoo
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hybrik.utils.presets import SimpleTransform3DSMPLX
from hamer.datasets.vitdet_dataset import ViTDetDataset
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from hybrik.models import builder
from torchvision import transforms as T
from hybrik.utils.config import update_config
from hamer.utils import recursive_to

## This code might not be correct, it is a work in progress
class KeypointExtractor:
    def __init__(self, video_name, source_dir, out_dir):
        self.video_name = video_name
        self.source_dir = source_dir
        self.out_dir = out_dir

    def extract(self):
        self._run_extraction()

    def _run_extraction(self):
        
        # Set up Detectron2 config
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cuda"
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)

        # SMPL-X/ hybrik-x model
        ckpt_path = './pretrained_models/hybrikx_hrnet.pth'
       
        smplx_cfg = update_config("./configs/smplx/256x192_hrnet_smplx_kid.yaml")

        smplx_cfg['MODEL']['EXTRA']['USE_KID'] = smplx_cfg['DATASET'].get('USE_KID', False)
        smplx_cfg['LOSS']['ELEMENTS']['USE_KID'] = smplx_cfg['DATASET'].get('USE_KID', False)

        
        bbox_3d_shape = [dim * 1e-3 for dim in getattr(smplx_cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))]
        
        dummy_set = edict({
            'joint_pairs_17': None,
            'joint_pairs_24': None,
            'joint_pairs_29': None,
            'bbox_3d_shape': bbox_3d_shape
        })

        transformation = SimpleTransform3DSMPLX(
            dummy_set,
            scale_factor=smplx_cfg.DATASET.SCALE_FACTOR,
            color_factor=smplx_cfg.DATASET.COLOR_FACTOR,
            occlusion=smplx_cfg.DATASET.OCCLUSION,
            input_size=smplx_cfg.MODEL.IMAGE_SIZE,
            output_size=smplx_cfg.MODEL.HEATMAP_SIZE,
            depth_dim=smplx_cfg.MODEL.EXTRA.DEPTH_DIM,
            bbox_3d_shape=bbox_3d_shape,
            rot=smplx_cfg.DATASET.ROT_FACTOR,
            sigma=smplx_cfg.MODEL.EXTRA.SIGMA,
            train=False,
            add_dpg=False,
            loss_type=smplx_cfg.LOSS['TYPE']
        )

        model = builder.build_sppe(smplx_cfg.MODEL).cuda()
        model.eval()

        download_models(CACHE_DIR_HAMER)
        hamer_model, hamer_model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        hamer_model = hamer_model.to('cuda')
        hamer_model.eval()

        print(f'Loading Hybrik-X model from {ckpt_path}...')

        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict.get('model', state_dict))

        out_path = os.path.join(self.out_dir, 'keypoints')

        os.makedirs(out_path, exist_ok=True)

        video_path = os.path.join(self.source_dir, self.video_name)

        res_keys = ['joints', 'expressions', 'camera', 'betas', 'phi', 'bbox', 'rot_mats', 'focal', 'vertices']
        res_db = {k: [] for k in res_keys}
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            with torch.no_grad():
                outputs = predictor(frame)
                boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

                if len(boxes) == 0:
                    continue

                x1, y1, x2, y2 = boxes[0].astype(int)
                h, w = frame.shape[:2]
                x1, x2 = np.clip([x1, x2], 0, w)
                y1, y2 = np.clip([y1, y2], 0, h)
                center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

                pose_input, box, img_center = transformation.test_transform(frame, boxes[0])
                pose_input = pose_input.to('cuda')[None,:,:,:] 

                pose_output = model(
                    pose_input,
                    flip_test=True,
                    bboxes=torch.from_numpy(np.array(box).to(pose_input.device).unsqueeze(0).float()),
                    img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()
                )
                is_right = []
                pose = transforms.matrix_to_axis_angle(pose_output.pred_theta_mat)
                left_hand = pose[:,75:120]
                right_hand = pose[:,120:]
                pose = pose[:,:75]
                keyp = left_hand
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    is_right.append(0)
                keyp = right_hand
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    is_right.append(1)

                right = np.stack(is_right)

                bbox = [
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    x2 - x1,
                    y2 - y1
                ]

                dataset = ViTDetDataset(hamer_model_cfg, frame, boxes, right, rescale_factor=2.0)
                print(f"Dataset length: {len(dataset)}")
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
                for batch in dataloader:
                    batch = recursive_to(batch, 'cuda')
                    with torch.no_grad():
                        out = model(batch)

                    pred_cam = out['pred_cam_t']
                    focal_length = out['focal_length']
                    joints = out['pred_keypoints_3d']
                    print(f"Joints shape: {joints.shape}")
                
                x = torch.cat(pose,joints, dim=1)
                print(f"Pose shape afte joints: {x.shape}")

                rot_mats = transforms.axis_angle_to_matrix(x)
                res_db['joints'].append(pose_output.pred_xyz_hybrik.view(1, -1, 3).detach().cpu().numpy()) ## i am not sure , hahahaha
                res_db['camera'].append(pred_cam.detach().cpu().numpy())
                res_db['rot_mats'].append(rot_mats.detach().cpu().numpy())
                res_db['betas'].append(pose_output.pred_shape_full.detach().cpu().numpy())
                res_db['phi'].append(pose_output.pred_phi.detach().cpu().numpy())
                res_db['bbox'].append(np.array(bbox))
                res_db['expressions'].append(pose_output.pred_expression.detach().cpu().numpy())
                res_db['focal'].append(focal_length.detach().cpu().numpy())
                res_db['vertices'].append(pose_output.pred_vertices.squeeze(1).detach().cpu().numpy())
            
        
        cap.release()
        # Save results


        base_name = os.path.splitext(self.video_name)[0]
        output_path = os.path.join(out_path, f"{base_name}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(res_db, f)

        print(f"Extraction complete. Results saved to: {out_path}")
