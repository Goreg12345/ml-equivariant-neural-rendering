import torch
import math
import cv2
import numpy as np
import os
import glob
import json 
import torchvision
import imageio

import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from torchvision.transforms import transforms

from misc.utils import full_rotation_angle_sequence, sine_squared_angle_sequence
irange = range

def make_grid_with_labels(tensor, labels=[''], nrow=8, limit=None, padding=2,
                          normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        labels (list):  ( [labels_1,labels_2,labels_3,...labels_n]) where labels is Bx1 vector of some labels
        limit ( int, optional): Limits number of images and labels to make grid of
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """

    if limit is not None:
        tensor = tensor[:limit, ::]
        labels = labels[:limit, ::]

    font = 1
    fontScale = 2
    color = (0, 0, 0)
    thickness = 1

    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0

    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            working_tensor = tensor[k]
            if labels is not None:
                org = (2, int(tensor[k].shape[1] * 0.3))
                working_image = cv2.UMat(
                    np.asarray(np.transpose(working_tensor.numpy(), (1, 2, 0)) * 255).astype('uint8'))
                image = cv2.putText(working_image, f'{str(labels[k])}', org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
                working_tensor = transforms.ToTensor()(image.get())
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(working_tensor)
            k = k + 1
    return grid

    def show(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


def plot_img_tensor(img, labels=[], save_file=None, nrow=4, figsize=None):
    """Helper function to plot image tensors.
    
    Args:
        img (torch.Tensor): Image or batch of images of shape 
            (batch_size, channels, height, width).
    """
    img_grid = make_grid_with_labels(img, labels, nrow=nrow)

    if figsize:
        plt.figure(figsize=figsize)

    plt.imshow(img_grid.cpu().numpy().transpose(1, 2, 0))
    
    if save_file:    
        plt.axis('off')
        plt.savefig(save_file, dpi=400, bbox_inches='tight')


def render_interpolated(model, scenes, time_step):
    lower_bound = math.floor(time_step)
    upper_bound = math.ceil(time_step)

    # Interpolate
    if lower_bound != upper_bound:
        scene_lower = scenes[0, lower_bound, ...] * (upper_bound - time_step) 
        scene_upper = scenes[0, upper_bound, ...] * (time_step - lower_bound)
        
        scene = scene_lower.add(scene_upper)

        # Use linear interpolation instead
        #scene = torch.lerp(scene_lower, scene_upper, time_step-lower_bound)
        
    # Return timestep without interpolation in case time_step is int
    else:
        scene = scenes[0, int(time_step), ...]
        
    scene = model.spherical_mask(scene).squeeze()
    scene = scene.unsqueeze(0)
 
    return model.render(scene)


def generate_scene_rep_images(scenes, time_step, nrow=4, clip=False, sigmoid=False, save_path='imgs/scene_rep'):
    rep_dim = scenes.shape[-1]
    tensor3d = scenes.squeeze()[time_step]
    
     # Clip values to be in range [0,1] for clearer output
    if clip:
        tensor3d = torch.clip(tensor3d, min=0., max=1.)
        
    if sigmoid:
        tensor3d = torch.sigmoid(tensor3d)

    tensor3d = tensor3d.permute(1,0,2,3)
    
    # create saving folder if it does not yet exist
    isExist = os.path.exists(save_path)

    if not isExist:
        os.makedirs(save_path)

    labels = ["" for t in range(rep_dim)]
    plot_img_tensor(
        tensor3d.detach(), 
        labels=labels, 
        save_file=f"{save_path}/scene_rep_vis_{str(time_step).zfill(2)}", 
        nrow=nrow
    )


def make_gif_scene_rep(frame_folder='imgs/scene_rep/'):
    sorted_names = sorted(glob.glob(f"{frame_folder}/*.png"))
    frames = [Image.open(image) for image in sorted_names]

    for i, img in enumerate(frames):
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 100)

        text = f"temporal: {i}"
        textwidth, textheight = draw.textsize(text, font)
        margin = 10
        x = 410
        y = 2380
        outline = (x - margin, y - margin, x + textwidth + margin, y + textheight + margin)

        draw.rectangle(outline, fill='black')
        draw.text((x, y), text, (255,255,255), font=font)

    frame_one = frames[0]
    frame_one.save(f"imgs/scene_renders.gif", format="GIF", append_images=frames,
               save_all=True, duration=80, loop=0)


def generate_novel_views(model, img_source, azimuth_source, elevation_source,
                         azimuth_shifts, elevation_shifts, temporal_dim):
    """Generates novel views of an image by inferring its scene representation,
    rotating it and rendering novel views. Returns a batch of images
    corresponding to the novel views.

    Args:
        model (models.neural_renderer.NeuralRenderer): Neural rendering model.
        img_source (torch.Tensor): Single image. Shape (channels, height, width).
        azimuth_source (torch.Tensor): Azimuth of source image. Shape (1,).
        elevation_source (torch.Tensor): Elevation of source image. Shape (1,).
        azimuth_shifts (torch.Tensor): Batch of angle shifts at which to
            generate novel views. Shape (num_views,).
        elevation_shifts (torch.Tensor): Batch of angle shifts at which to
            generate novel views. Shape (num_views,).
    """
    # No need to calculate gradients
    with torch.no_grad():
        num_views = len(azimuth_shifts)
        # Create scene representation
        scenes = model.inverse_render(img_source)
        scenes = scenes.reshape(1, model.temporal_channels, int(scenes.shape[1] / model.temporal_channels), *scenes.shape[2:])
        # Pick temporal dimension
        scenes = scenes[:,temporal_dim,...]
        # Apply spherical_maks
        scenes = model.spherical_mask(scenes).squeeze()
        scenes = scenes.unsqueeze(0)
        # Copy scene for each target view
        scenes_batch = scenes.repeat(num_views, 1, 1, 1, 1)
        # Batchify azimuth and elevation source
        azimuth_source_batch = azimuth_source.repeat(num_views)
        elevation_source_batch = elevation_source.repeat(num_views)
        # Calculate azimuth and elevation targets
        azimuth_target = azimuth_source_batch + azimuth_shifts
        elevation_target = elevation_source_batch + elevation_shifts
        # Rotate scenes
        rotated = model.rotate_source_to_target(scenes_batch, azimuth_source_batch,
                                                elevation_source_batch,
                                                azimuth_target, elevation_target)

    # Render images
    return model.render(rotated).detach()


def save_img_sequence_as_gif(img_sequence, filename, nrow=4):
    """Given a sequence of images as tensors, saves a gif of the images.
    If images are in batches, they are converted to a grid before being
    saved.

    Args:
        img_sequence (list of torch.Tensor): List of images. Tensors should
            have shape either (batch_size, channels, height, width) or shape
            (channels, height, width). If there is a batch dimension, images
            will be converted to a grid.
        filename (string): Path where gif will be saved. Should end in '.gif'.
        nrow (int): Number of rows in image grid, if image has batch dimension.
    """
    img_grid_sequence = []
    for img in img_sequence:
        if len(img.shape) == 4:
            img_grid = torchvision.utils.make_grid(img, nrow=nrow)
        else:
            img_grid = img

        img_grid = (img_grid * 255.).byte().cpu().numpy().transpose(1, 2, 0)
        img_grid_sequence.append(img_grid)
    
    # Save gif
    imageio.mimsave(filename, img_grid_sequence)

def batch_generate_novel_views(model, imgs_source, azimuth_source,
                               elevation_source, azimuth_shifts,
                               elevation_shifts, temporal_dims):
    """Generates novel views for a batch of images. Returns a list of batches of
    images, where each item in the list corresponds to a novel view for all
    images.

    Args:
        model (models.neural_renderer.NeuralRenderer): Neural rendering model.
        imgs_source (torch.Tensor): Source images. Shape (batch_size, channels, height, width).
        azimuth_source (torch.Tensor): Azimuth of source. Shape (batch_size,).
        elevation_source (torch.Tensor): Elevation of source. Shape (batch_size,).
        azimuth_shifts (torch.Tensor): Batch of angle shifts at which to generate
            novel views. Shape (num_views,).
        elevation_shifts (torch.Tensor): Batch of angle shifts at which to
            generate novel views. Shape (num_views,).
    """
    num_imgs = imgs_source.shape[0]
    num_views = azimuth_shifts.shape[0]
    
    # Initialize novel views, i.e. a list of length num_views with each item
    all_novel_views = [torch.zeros_like(imgs_source) for _ in range(num_views*len(temporal_dims))]

    for temporal_dim in temporal_dims:
        for i in range(num_imgs):
            # Generate novel views for single image
            novel_views = generate_novel_views(model, imgs_source,
                                               azimuth_source[i:i+1],
                                               elevation_source[i:i+1],
                                               azimuth_shifts, elevation_shifts, temporal_dim).cpu()
            # Add to list of all novel_views
            for j in range(num_views):
                all_novel_views[j + (num_views * temporal_dim)][i] = novel_views[j]

    return all_novel_views

def make_gif_of_dataset(frame_folder, save_name):
    sorted_names = sorted(glob.glob(f"{frame_folder}/*.png"))
    frames = [Image.open(image) for image in sorted_names]
    
    # Add text to image
    params_dict = json.load(open(frame_folder + '/render_params.json'))
    for i, img in enumerate(frames):
        draw = ImageDraw.Draw(img)
        params = params_dict[sorted_names[i].strip(frame_folder).strip('.png')]
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 9)
        draw.text((0, 0), f" azimuth: {params['azimuth']:.4f}\n elevation: {params['elevation']:.4f}\n temporal: {params['temporal']}", (0,0,0), font=font)
        
    frame_one = frames[0]
    frame_one.save(save_name, format="GIF", append_images=frames,
               save_all=True, duration=80, loop=0)
