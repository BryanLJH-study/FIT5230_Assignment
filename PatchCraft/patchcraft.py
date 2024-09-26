import os
import random as rand
from datetime import datetime

import numpy as np
import cv2
from scipy.ndimage import rotate

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim


def random_patches(img, num_patches=192, patch_size=32):
  """
  Returns a list of randomly cropped patches from an image.
  """
  height, width, _ = img.shape
  patches = []
  for _ in range(num_patches):
    x = rand.randint(0, width - patch_size)
    y = rand.randint(0, height - patch_size)
    patch = img[y:y + patch_size, x:x + patch_size]
    patches.append(patch)

  return patches


def combine_patches(patches, rows=8, cols=8):
  """
  Combines the image patches into a single image collage of gridsize=(rows,cols)
  """
  # Preparing image canvas
  patch_height, patch_width, _ = patches[0].shape
  grid_height = patch_height * rows
  grid_width = patch_width * cols
  grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

  # Fill image canvas patch by patch
  for i, patch in enumerate(patches[:rows * cols]):
    row = i // cols
    col = i % cols
    y = row * patch_height
    x = col * patch_width
    grid_img[y:y + patch_height, x:x + patch_width] = patch

  return grid_img


def texture_diversity(patch):
  """
  Measures texture diversity of a patch (square) by pixel fluctuation degree.
  Texture diversity score = sum of residuals of four directions (horizontal, vertical, diagonal, counter diagonal)

  Args:
    patch: A M x M numpy array representing the patch.

  Returns:
    The texture diversity score of the patch.

  Note: numpy is arrays have shape (height, width), which is why i & j are swapped from formula
  """
  patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
  patch = patch.astype(int)

  m, _ = patch.shape
  residual_sum = 0

  # Horizontal direction
  for i in range(m-1):
    for j in range(m):
      residual_sum += abs(patch[j][i] - patch[j][i+1])

  # Vertical direction
  for i in range(m):
    for j in range(m-1):
      residual_sum += abs(patch[j][i] - patch[j+1][i])

  # Diagonal & Counter-diagonal direction
  for i in range(m - 1):
    for j in range(m - 1):
      residual_sum += abs(patch[j][i] - patch[j+1][i+1])
      residual_sum += abs(patch[j+1][i] - patch[j][i+1])

  return residual_sum


def extract_rich_and_poor_textures(patches):
  """
  Sorts patches based on texture diversity and divides them into rich and poor texture patches.

  Args:
    patches: A list of image patches (numpy arrays).

  Returns:
    A tuple containing two lists: rich_patches and poor_patches.
  """
  # Calculate texture diversity scores for each patch
  diversity_scores = [texture_diversity(patch) for patch in patches]

  # Sort patches based on their texture diversity scores
  sorted_patches_with_scores = sorted(zip(patches, diversity_scores), key=lambda x: x[1])

  # Divide patches into  poor (bottom 33%) and rich (top 33%) texture patches
  num_patches = len(patches)
  poor_patches = [patch for patch, _ in sorted_patches_with_scores[:num_patches // 3]]
  rich_patches = [patch for patch, _ in sorted_patches_with_scores[2 * num_patches // 3:]]

  return poor_patches, rich_patches


def smash_and_reconstruct(img):
  """
  Performs entire Smash & Reconstruct process on an image.

  Args:
    img: A numpy array representing the input image.

  Returns:
    A tuple containing two numpy arrays for poor and rich texture regions respectively.
  """
  # Extract rich & poor texture patches
  patches = random_patches(img, num_patches=192, patch_size=32)
  poor_patches, rich_patches = extract_rich_and_poor_textures(patches)

  # Recombine patches into images
  poor_img = combine_patches(poor_patches, rows=8, cols=8)
  rich_img = combine_patches(rich_patches, rows=8, cols=8)

  return poor_img, rich_img


# Base Filters
filter_a = np.array([[ 0,  0,  0,  0,  0],
                     [ 0,  0,  1,  0,  0],
                     [ 0,  0, -1,  0,  0],
                     [ 0,  0,  0,  0,  0],
                     [ 0,  0,  0,  0,  0]])

filter_b = np.array([[ 0,  0, -1,  0,  0],
                     [ 0,  0,  3,  0,  0],
                     [ 0,  0, -3,  0,  0],
                     [ 0,  0,  1,  0,  0],
                     [ 0,  0,  0,  0,  0]])

filter_c = np.array([[ 0,  0,  0,  0,  0],
                     [ 0,  0,  1,  0,  0],
                     [ 0,  0, -2,  0,  0],
                     [ 0,  0,  1,  0,  0],
                     [ 0,  0,  0,  0,  0]])

filter_d = np.array([[ 0,  0,  0,  0,  0],
                     [ 0, -1,  2, -1,  0],
                     [ 0,  2, -4,  2,  0],
                     [ 0,  0,  0,  0,  0],
                     [ 0,  0,  0,  0,  0]])

filter_e = np.array([[-1,  2, -2,  2, -1],
                     [ 2, -6,  8, -6,  2],
                     [-2,  8,-12,  8, -2],
                     [ 0,  0,  0,  0,  0],
                     [ 0,  0,  0,  0,  0]])

filter_f = np.array([[ 0,  0,  0,  0,  0],
                     [ 0, -1,  2, -1,  0],
                     [ 0,  2, -4,  2,  0],
                     [ 0, -1,  2, -1,  0],
                     [ 0,  0,  0,  0,  0]])

filter_g = np.array([[-1,  2, -2,  2, -1],
                     [ 2, -6,  8, -6,  2],
                     [-2,  8,-12,  8, -2],
                     [ 2, -6,  8, -6,  2],
                     [-1,  2, -2,  2, -1]])


def filter_variations(base_filters, directions_list):
  """
  Generates a list of filters by rotating the base filters in various directions.

  Args:
    base_filters: A list of numpy arrays representing the base filters.
    directions_list: A list of lists representing the directions for rotation.
      - 0 = No rotation,
      - 1 = 45 degrees,
      - 2 = 90 degrees,
      - 3 = 135 degrees,
      - 4 = 180 degrees,
      - 5 = 225 degrees,
      - 6 = 270 degrees,
      - 7 = 315 degrees.

  Returns:
    A list of rotated filters.
  """
  angles= [-i*45 for i in range(8)]
  filters = []

  # Get each filter for each base fiter, for each rotation
  for base_filter, directions in zip(base_filters, directions_list):
    for d in directions:
      filter = rotate(base_filter, angle=angles[d], reshape=False)
      filters.append(filter)
  return filters


def apply_filters(img, base_filters, directions_list):
  """
  Applies a list of filters to an image.

  Args:
    img: A numpy array representing the input image.
    base_filters: A list of numpy arrays representing the base filters.
    directions_list: A list of lists representing the directions for rotation.

  Returns:
    The extracted features in a 3D numpy array of shape (C, H, W).
  """
  result = []

  # Obtain complete filter list
  filters = filter_variations(base_filters, directions_list)

  # For each filter, convert image to greyscale and apply filter
  for filter in filters:
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.filter2D(src=img_grey, kernel=filter, ddepth=-1)
    result.append(filtered_img)

  # Convert list of filters into a single 3d numpy array
  result = np.dstack(result)
  result = np.rollaxis(result,-1) # shape (H, W, C) -> (C, H, W) for pytorch

  return result


def apply_high_pass_filters(img):
  """
  Applies all 30 high pass filters to an image.

  Args:
    img: A numpy array representing the input image.

  Returns:
    The extracted features in a 3D numpy array of shape (C, H, W).
  """
  # Define high pass fiters
  base_filters = [filter_a, filter_b, filter_c, filter_d, filter_e, filter_f, filter_g]

  directions_list = [[i for i in range(8)],
                    [i for i in range(8)],
                    [1,2,3,4],
                    [0,2,4,6],
                    [0,2,4,6],
                    [0],
                    [0]]

  # Apply fiterss
  img_filtered = apply_filters(img, base_filters, directions_list)

  return img_filtered


def preprocess(img):
  """
  Wrapper function for image preprocessing.

  Args:
    img: The input image

  Returns:
    A tuple containing two numpy arrays of shape (C, H, W) for poor and rich texture regions respectively.
  """
  # Convert input image into numpy array
  img = np.array(img)

  # Extract poor and rich texture regions, and apply high pass filters
  poor, rich = smash_and_reconstruct(img)
  poor = apply_high_pass_filters(poor)
  rich = apply_high_pass_filters(rich)

  return poor, rich



class PatchCraftModel(nn.Module):
  def __init__(self):
     super(PatchCraftModel, self).__init__()

     # Convolution block learns inter-pixel correlation from the input high-pass filtered images
     self.hpv_extraction_block = nn.Sequential(
        nn.Conv2d(30, 1, kernel_size=3, bias=False),
        nn.BatchNorm2d(1),
        nn.Hardtanh(),
     )
     # First convolution blocks of classifier
     self.double_conv_block_1 = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, bias=False), # Bias = False because it is redundant when followed by BatchNorm
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
     )
     # Convolution blocks
     self.double_conv_block = nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=3, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
     )
     # Pooling layers
     self.AvgPool = nn.AvgPool2d(kernel_size=2)
     self.AdpAvgPool = nn.AdaptiveAvgPool2d((1,1))

     # Flattening & Fully Connected Layers
     self.flatten = nn.Flatten()
     self.fc = nn.Linear(32, 1)


  def forward(self, poor_texture, rich_texture):
    # Extract inter-pixel correlation from high pass fiters processed poor and rich texture regions
    poor_features = self.hpv_extraction_block(poor_texture)
    rich_features = self.hpv_extraction_block(rich_texture)

    # Fingerprint: inter-pixel correlation contrast between rich and poor texture regions
    fingerprint = rich_features - poor_features

    # Classifier
    x = self.double_conv_block_1(fingerprint)

    for _ in range(3):
      x = self.double_conv_block(x)
      x = self.AvgPool(x)

    x = self.double_conv_block(x)
    x = self.AdpAvgPool(x)

    x = self.flatten(x)
    x = self.fc(x)

    return x
  

class PatchCraftDetector:
  def __init__(self, model_path, device) -> None:
    self.device = device
    self.model =  PatchCraftModel().to(device)
    self.model.load_state_dict(torch.load(model_path))
    self.model.eval()


  def classify(self, img):
    poor, rich = preprocess(img)
    poor = torch.from_numpy(poor).unsqueeze(0).float().to(self.device)
    rich = torch.from_numpy(rich).unsqueeze(0).float().to(self.device)
    output = self.model(poor, rich)
    prob = torch.sigmoid(output).item() 

    return prob
