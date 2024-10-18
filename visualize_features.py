# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copied from DinoV1
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
import tqdm
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import math

from dinov2 import utils
from dinov2.models import vision_transformer as vits

class Visualizer():
    def __init__(self, model, patch_size, output_dir, threshold=None, device="cpu"):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.w_featmap = img.shape[-2] // patch_size[0]
        self.h_featmap = img.shape[-1] // patch_size[1]
        self.threshold = threshold

    def _apply_mask(self, image, mask, color, alpha=0.5):
        for c in range(3):
            image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        return image


    def _random_colors(self, N, bright=True):
        """
        Generate random colors.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors


    def _display_instances(self, image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
        fig = plt.figure(figsize=figsize, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax = plt.gca()

        N = 1
        mask = mask[None, :, :]
        # Generate random colors
        colors = self._random_colors(N)

        # Show area outside image boundaries.
        height, width = image.shape[:2]
        margin = 0
        ax.set_ylim(height + margin, -margin)
        ax.set_xlim(-margin, width + margin)
        ax.axis('off')
        masked_image = image.astype(np.uint32).copy()
        for i in range(N):
            color = colors[i]
            _mask = mask[i]
            if blur:
                _mask = cv2.blur(_mask,(10,10))
            # Mask
            masked_image = self._apply_mask(masked_image, _mask, color, alpha)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            if contour:
                padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
                padded_mask[1:-1, 1:-1] = _mask
                contours = find_contours(padded_mask, 0.5)
                for verts in contours:
                    # Subtract the padding and flip (y, x) to (x, y)
                    verts = np.fliplr(verts) - 1
                    p = Polygon(verts, facecolor="none", edgecolor=color)
                    ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8), aspect='auto')
        fig.savefig(fname)
        print(f"{fname} saved.")
        return
    
    def visualize_attn(self, img):
        attentions = self.model.get_last_self_attention(img.to(device))
        nh = attentions.shape[1] # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        if self.threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - self.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=self.patch_size, mode="nearest")[0].cpu().numpy()

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=self.patch_size, mode="nearest")[0].cpu()
        
        # save attentions heatmaps
        os.makedirs(self.output_dir, exist_ok=True)
        torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(self.output_dir, "img.png"))
        for j in range(nh):
            fname = os.path.join(self.output_dir, "attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            print(f"{fname} saved.")

        if self.threshold is not None:
            image = skimage.io.imread(os.path.join(self.output_dir, "img.png"))
            for j in range(nh):
                self._display_instances(image, th_attn[j], fname=os.path.join(self.output_dir, "mask_th" + str(self.threshold) + "_head" + str(j) +".png"), blur=False)

    def visualize_pos(self, img):
        pos_embed = self.model.get_first_pos_embedding(img.to(device)).to(device)
        # pos_embed = pos_embed[0, :-1, :]
        pos_embed = pos_embed[:, 1:, :]
        pos_embed = pos_embed.reshape(1, self.w_featmap, self.h_featmap, -1)

        scores = torch.empty((self.w_featmap, self.h_featmap, self.w_featmap, self.h_featmap)).cpu()
        for i in range(0, self.w_featmap):
            for j in range(0, self.h_featmap):
                curr_embed = pos_embed[0, i, j, :][None][None]
                score = torch.cosine_similarity(curr_embed, pos_embed[0, :, :, :], dim=-1)
                scores[i, j, :, :] = score.cpu()
            
        self.plot_pos_embed(scores)
    
    def plot_pos_embed(self, data):
        # Set up the plot
        fig = plt.figure(figsize=(10, 10))

        # Reshape the scores to a square
        x_dim = math.ceil(math.sqrt(data.shape[0]))
        x_dim_sqrd = x_dim**2
        pad_amount = x_dim_sqrd-data.shape[0]
        padding1 = torch.full((pad_amount,1,data.shape[0],1), -1, dtype=torch.float32).cpu()
        padding2 = torch.full((x_dim_sqrd,1,pad_amount,1), -1, dtype=torch.float32).cpu()
        result = torch.cat((data, (padding1)), dim=0)
        result = torch.cat((result, padding2), dim=2)
        square_data = result.reshape(x_dim, x_dim, x_dim, x_dim)
        grid_shape = (x_dim, x_dim)

        # Create the main grid
        main_grid = fig.add_gridspec(grid_shape[0], grid_shape[1], wspace=0.1, hspace=0.1)

        for i in tqdm.trange(grid_shape[0], desc="Creating subplots"):
            for j in range(grid_shape[1]):
                sub_ax = fig.add_subplot(main_grid[i, j])                
                im = sub_ax.imshow(square_data[i, j], cmap='viridis', vmin=-1, vmax=1, interpolation='nearest')
                
                sub_ax.set_xticks([])
                sub_ax.set_yticks([])
                
                # Add borders
                for spine in sub_ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('0.0')
            
                # label y 
                if main_grid[i,j].is_first_col():
                    sub_ax.set_ylabel(i, fontsize = 16, rotation=0, labelpad=10)

                # label x 
                if main_grid[i,j].is_last_row():
                    sub_ax.set_xlabel(j, fontsize = 16, labelpad=10)

        # Add colorbar to the right of the subplots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Cosine similarity', rotation=270, labelpad=20)
        cbar.set_ticks([-1,1])

        # Set overall title and labels
        fig.suptitle("Position embedding similarity", fontsize=20, y=0.95)
        fig.text(0.5, 0.04, 'Patch index', ha='center', va='center', fontsize=18)
        fig.text(0.04, 0.5, 'Patch index offset', ha='center', va='center', rotation='vertical', fontsize=18)

        fig.savefig(os.path.join(self.output_dir, "pos_embed.png"))
        plt.close()

    def visualize_feats(self, task, img):
        if task == 'attn':
            self.visualize_attn(img)
        elif task == 'pos':
            self.visualize_pos(img)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=1, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--data_path", default=None, type=str, help="Path of the data to load.")
    parser.add_argument("--data_size", default=(186,240), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument("--feat_type", choices=['pos', 'attn'], default='pos', help='Feature type to visualize.')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, img_size=args.data_size)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        sys.exit(1)

    # open image
    if args.data_path is None:
        print(f"No data has been provided.")
        sys.exit(1)
    elif os.path.isfile(args.data_path):
        img = np.load(args.data_path)
    else:
        print(f"Provided image path {args.data_path} is non valid.")
        sys.exit(1)

    img = torch.from_numpy(img).permute(2,0,1)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0).float()

    w_featmap = img.shape[-3] // args.patch_size
    h_featmap = img.shape[-2] // args.patch_size
    
    visualizer = Visualizer(
        model, 
        patch_size=(args.patch_size,240), 
        output_dir=args.output_dir, 
        threshold=args.threshold, 
        device=device
    )

    visualizer.visualize_feats(args.feat_type, img)