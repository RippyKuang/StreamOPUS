import os
import mmcv
import numpy as np
import torch
import math
import os.path as osp
from tqdm import tqdm
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from torch.utils.data import DataLoader
from models.utils import sparse2dense
from .old_metrics import Metric_mIoU

@DATASETS.register_module()
class NuScenesOccDataset(NuScenesDataset):    
    def __init__(self, seq_mode=False, seq_split_num=1, *args, **kwargs):
        super().__init__(filter_empty_gt=False, *args, **kwargs)
        self.seq_mode = seq_mode
        self.data_infos = self.load_annotations(self.ann_file)
        if seq_mode:
            self.seq_split_num = seq_split_num
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['cam_sweeps']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    
    def collect_cam_sweeps(self, index, into_past=150, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['cam_sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['cam_sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def collect_lidar_sweeps(self, index, into_past=20, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['lidar_sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        last_timestamp = self.data_infos[index]['timestamp']
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['lidar_sweeps'][::-1]
            if curr_sweeps[0]['timestamp'] == last_timestamp:
                curr_sweeps = curr_sweeps[1:]
            all_sweeps_next.extend(curr_sweeps)
            curr_index = curr_index + 1
            last_timestamp = all_sweeps_next[-1]['timestamp']

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]
      
        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix
        ego2lidar = transform_matrix(
            lidar2ego_translation, Quaternion(lidar2ego_rotation), inverse=True)
        ego2global = transform_matrix(
            ego2global_translation, Quaternion(ego2global_rotation), inverse=False)
        input_dict = dict(
            sample_idx=info['token'],
            scene_name=info['scene_name'],
            timestamp=info['timestamp'] / 1e6,
            prev_exists= not (index == 0 or self.flag[index - 1] != self.flag[index]) if not self.test_mode else None,
            lidar2global=ego2global @ np.linalg.inv(ego2lidar),
            ego2lidar=ego2lidar,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
        )

        if self.modality['use_lidar']:
            lidar_sweeps_prev, lidar_sweeps_next = self.collect_lidar_sweeps(index)
            input_dict.update(dict(
                pts_filename=info['lidar_path'],
                lidar_sweeps={'prev': lidar_sweeps_prev, 'next': lidar_sweeps_next},
            ))

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []

            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            cam_sweeps_prev, cam_sweeps_next = self.collect_cam_sweeps(index)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                cam_sweeps={'prev': cam_sweeps_prev, 'next': cam_sweeps_next},
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
    
    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        results_dict = {}
        results_dict.update(
            self.eval_miou(occ_results, runner=runner, show_dir=show_dir, **eval_kwargs))
        results_dict.update(
            self.eval_riou(occ_results, runner=runner, show_dir=show_dir, **eval_kwargs))
        return results_dict

    def eval_miou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts = []
        occ_preds = []
        lidar_origins = []

        print('\nStarting Evaluation...')
        metric = Metric_mIoU(use_image_mask=True)

        from tqdm import tqdm
        for i in tqdm(range(len(occ_results))):
            result_dict = occ_results[i]
            info = self.get_data_info(i)
            token = info['sample_idx']
            scene_name = info['scene_name']
            occ_root = 'data/nuscenes/gts/'
            occ_file = osp.join(occ_root, scene_name, token, 'labels.npz')
            occ_infos = np.load(occ_file)

            occ_labels = occ_infos['semantics']
            mask_lidar = occ_infos['mask_lidar'].astype(np.bool_)
            mask_camera = occ_infos['mask_camera'].astype(np.bool_)

            occ_pred, _ = sparse2dense(
                result_dict['occ_loc'],
                result_dict['sem_pred'],
                dense_shape=occ_labels.shape,
                empty_value=17)
            
            metric.add_batch(occ_pred, occ_labels, mask_lidar, mask_camera)
        
        return {'mIoU': metric.count_miou()}
    
    def eval_riou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts = []
        occ_preds = []
        lidar_origins = []

        print('\nStarting Evaluation...')

        from .ray_metrics import main as calc_rayiou
        from .ego_pose_dataset import EgoPoseDataset

        data_loader = DataLoader(
            EgoPoseDataset(self.data_infos),
            batch_size=1,
            shuffle=False,
            num_workers=8
        )
        
        sample_tokens = [info['token'] for info in self.data_infos]

        for i, batch in enumerate(data_loader):
            token = batch[0][0]
            output_origin = batch[1]
            
            data_id = sample_tokens.index(token)
            info = self.data_infos[data_id]

            token = info['token']
            scene_name = info['scene_name']
            occ_root = 'data/nuscenes/gts/'
            occ_file = osp.join(occ_root, scene_name, token, 'labels.npz')
            occ_infos = np.load(occ_file)
            gt_semantics = occ_infos['semantics']

            occ_pred = occ_results[data_id]
            sem_pred = torch.from_numpy(occ_pred['sem_pred'])  # [B, N]
            occ_loc = torch.from_numpy(occ_pred['occ_loc'].astype(np.int64))  # [B, N, 3]
            
            occ_size = list(gt_semantics.shape)
            dense_sem_pred, _ = sparse2dense(occ_loc, sem_pred, dense_shape=occ_size, empty_value=17)
            dense_sem_pred = dense_sem_pred.squeeze(0).numpy()

            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            occ_preds.append(dense_sem_pred)
        
        return calc_rayiou(occ_preds, occ_gts, lidar_origins)

    def format_results(self, occ_results,submission_prefix,**kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path=os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path,occ_pred.astype(np.uint8))
        print('\nFinished.')