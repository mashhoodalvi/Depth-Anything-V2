import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
import random


def hypersim_distance_to_depth(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth


class Hypersim(Dataset):
    def __init__(self, filelist_path, mode, relative_path, size=(518, 518), max_depth = 20,
                  min_depth = 0.001, invert_prior: int = 0, normalize_prior: int = 1, add_mask: int = 1, gt_prior: int = 0):
        
        self.mode = mode
        self.size = size
        self.relative_path = relative_path
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.invert_prior = invert_prior
        self.normalize_prior = normalize_prior
        self.add_mask = add_mask
        self.gt_prior = gt_prior
        self.epoch = 0

        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
        
    def __getitem__(self, item):

        if self.mode == "val":
            seed = item  
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        img_path = os.path.join(self.relative_path, self.filelist[item].split(' ')[0])
        depth_path = os.path.join(self.relative_path, self.filelist[item].split(' ')[1])
        label_path = os.path.join(self.relative_path, self.filelist[item].split(' ')[1].replace("depth_meters", "semantic"))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth_fd = h5py.File(depth_path, "r")
        distance_meters = np.array(depth_fd['dataset'])  #alternative depth_fd['dataset'][:]
        depth = hypersim_distance_to_depth(distance_meters)


        if self.mode == "train":
            depth *= random.uniform(0.5, 2) 
            max_points_train = max(1000 - self.epoch * 50, 100)


        label_fd = h5py.File(label_path, "r")
        label = np.array(label_fd['dataset']) 
        
        sample = self.transform({'image': image, 'depth': depth, 'label': label})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['valid_mask'] = (torch.isnan(sample['depth']) == 0)
        sample['depth'][sample['valid_mask'] == 0] = 0
        if self.mode == "train":
            sample['prior'] = self.create_prior(sample['depth'], torch.from_numpy(sample['label']), max_points=max_points_train)
        else:
            depth_resized = torch.from_numpy(sample['depth_resized'])
            sample['prior'] = self.create_prior(depth_resized, torch.from_numpy(sample['label']))

        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def create_prior(
        self,
        depth: torch.Tensor,
        label: torch.Tensor,
        #num_samples: int = 500,
        noise_std: float = 0.01,
        outlier_prob: float = 0.1,
        shift_max: int = 0,
        max_points: int = 100
    ):
        """
        Create a sparse noisy depth prior from GT depth.
        depth: (H, W) tensor
        returns: (H, W) tensor
        """

        device = depth.device
        H, W = depth.shape
        prior = torch.zeros((H, W), device=device)


        # valid depth mask
        valid_mask = (depth > 0) & ~(torch.isnan(depth)) & (depth < self.max_depth)
        valid_indices = valid_mask.nonzero(as_tuple=False)

        if len(valid_indices) == 0:
            return prior
        
        if random.random() < 0.7:  #sparse prior
        # sample valid pixels
            num_samples = np.random.randint(5, max_points)
            #num_samples = 500
            num_samples = min(num_samples, len(valid_indices))
            perm = torch.randperm(len(valid_indices), device=device)[:num_samples]
            sampled = valid_indices[perm]  # (N, 2)

            rows, cols = sampled[:, 0], sampled[:, 1]

            row_shift = torch.randint(-shift_max, shift_max + 1, (num_samples,), device=device)
            col_shift = torch.randint(-shift_max, shift_max + 1, (num_samples,), device=device)

            shifted_rows = torch.clamp(rows + row_shift, 0, H - 1)
            shifted_cols = torch.clamp(cols + col_shift, 0, W - 1)

            # sample values from shifted positions
            sampled_values = depth[shifted_rows, shifted_cols]

            # add gaussian noise
            sampled_values = sampled_values + torch.randn_like(sampled_values) * noise_std

            # add outliers
            #if num_samples >= 20:
            outlier_mask = torch.rand_like(sampled_values) < outlier_prob
            sampled_values[outlier_mask] += torch.randn_like(sampled_values[outlier_mask]) * noise_std * 20

            # scatter into prior
            prior[rows, cols] = sampled_values
        else:
            # ---- rectangle-based prior ----
            # number of rectangles (bias towards 1)
            probs = torch.tensor([0.2, 0.3, 0.5], device=device)
            prior_number = torch.multinomial(probs, 1).item() + 1  # 1..3

            # rectangle size distribution (biased around mean)
            if self.mode == "train" and (self.epoch < 15):
                scalar = max(1/(1+self.epoch * 0.5), 0.15)
                min_size = int(scalar * 0.4 * min(H, W))
                max_size = int(scalar * 0.8 * min(H, W))
                mean_size = int(scalar * 0.6 * min(H, W))
                std_size = int(scalar * 0.1 * min(H, W))
            else:
                min_size = int(0.06 * min(H, W))
                max_size = int(0.10 * min(H, W))
                mean_size = int(0.08 * min(H, W))
                std_size = int(0.01 * min(H, W))

            shift_max = 0

            for _ in range(prior_number):
                # sample rectangle size (clipped normal)
                h = int(torch.normal(mean_size, std_size, size=(1,), device=device).clamp(min_size, max_size))
                w = int(torch.normal(mean_size, std_size, size=(1,), device=device).clamp(min_size, max_size))

                # sample center (safe bounds incl. misalignment)
                cx = torch.randint(h // 2 + shift_max, H - h // 2 - shift_max, (1,), device=device).item()
                cy = torch.randint(w // 2 + shift_max, W - w // 2 - shift_max, (1,), device=device).item()

                x0, x1 = cx - h // 2, cx + h // 2
                y0, y1 = cy - w // 2, cy + w // 2

                depth_patch = depth[x0:x1, y0:y1]
                label_patch = label[x0:x1, y0:y1]

                # ignore invalid depth
                valid = depth_patch > 0
                if valid.sum() == 0:
                    continue

                # dominant semantic label
                labels, counts = torch.unique(label_patch[valid], return_counts=True)
                dominant_label = labels[counts.argmax()]

                label_mask = label_patch == dominant_label
                final_mask = valid if random.random() < 0.8 else (valid & label_mask)  #randomly use label

                if final_mask.sum() == 0:
                    continue

                # random misalignment
                dx = torch.randint(-shift_max, shift_max + 1, (1,), device=device).item()
                dy = torch.randint(-shift_max, shift_max + 1, (1,), device=device).item()

                tx0 = max(0, min(H, x0 + dx))
                tx1 = max(0, min(H, x1 + dx))
                ty0 = max(0, min(W, y0 + dy))
                ty1 = max(0, min(W, y1 + dy))

                sx0 = tx0 - (x0 + dx)
                sx1 = sx0 + (tx1 - tx0)
                sy0 = ty0 - (y0 + dy)
                sy1 = sy0 + (ty1 - ty0)

                #prior[tx0:tx1, ty0:ty1][final_mask[sx0:sx1, sy0:sy1]] = depth_patch[sx0:sx1, sy0:sy1][final_mask[sx0:sx1, sy0:sy1]]
                prior[tx0:tx1, ty0:ty1][final_mask] = depth_patch[final_mask]

        # if self.mode == "train":
        #     prior = depth.clone()
        #     prior[torch.isnan(prior)] = 0


        if self.gt_prior == 1:
            prior = depth.clone()
            prior[torch.isnan(prior)] = 0

        if (prior > 0).sum() == 0:
            print("sampling did not work")

        if prior.ndim == 2: # ensure 1, H, W
            prior = prior.unsqueeze(0)

        if self.add_mask == 1:
            prior_mask = (prior > 0).float()
            prior = torch.concat([prior, prior_mask], dim=0)  # 2, H, W

        if self.invert_prior == 1:
            prior = self.inverse_depth(prior)

        if self.normalize_prior == 1:
            prior = prior /self.max_depth

        # if random.random() < 0.05:
        #     prior.zero_()

        return prior 

    def inverse_depth(self, prior):
        disparity = torch.zeros_like(prior)
        non_negtive_mask = prior > 0
        disparity[non_negtive_mask] = 1.0 / prior[non_negtive_mask]
        return disparity


    def __len__(self):
        return len(self.filelist)