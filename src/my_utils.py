import os
import colorsys
import numpy as np
from sklearn.linear_model import RANSACRegressor
from pathlib import Path
from typing import Tuple, List, Union
Numeric = Union[int, float, np.number] # create generic numeric type (excluding complex numbers)


def find_files_by_ext(dir:Union[str, Path], extensions:List[str]=['png', 'jpg', 'jpeg']) -> List[Path]:
    """Find files in a given directory by a list of filenames. 
    This function DOES NOT look for files in subdirectories. 

    Args:
        dir (Union[str, Path]): Path to directory as string or pathlib.Path
        extensions (List[str], optional): What extensions to look for. Defaults to ['png', 'jpg', 'jpeg'].

    Returns:
        List[Path]: Files
    """    
    files: List[Path] = []
    for ext in extensions:
        ext = ext.split('.')[-1] # remove potential '.' from extension string
        files += list(Path(dir).glob(f'*.{ext}'))
    return files


def rgb_to_cv2_hsv(rgb: List[Numeric], max_val=255) -> Tuple[Numeric]:
    """Converts RGB values into the OpenCV HSV format (0-180, 0-255, 0-255)

    Args:
        rgb (List[Numeric]): RGB values
        max_val (int, optional): Range of the input RGB values. Defaults to 255.

    Returns:
        Tuple[Numeric]: Color in OpenCV HSV format (0-180, 0-255, 0-255)
    """    
    #get rgb percentage: range (0-1, 0-1, 0-1 )
    rgb_percentages = [c/max_val for c in rgb]

    # colorsys works on normalized color coordinates (0.0-1.0)
    hsv_percentage=colorsys.rgb_to_hsv(*rgb_percentages)

    #convert to opencv hsv format: range (0-180, 0-255, 0-255)
    hsv = [c*range for c, range in zip(hsv_percentage, (180, 255, 255))]
    
    return hsv


def iterative_RANSAC(mask: np.array, min_samples: int, min_inliers: int, residual_threshold: Numeric, max_trials: int, max_lines:int=8):
    """Iteratively use RANSAC to find lines in a mask. 

    Args:
        mask (np.array): Input mask
        min_samples (int): Number of samples to draw (use >=2 for detection of lines)
        min_inliers (int): Number of inliers to select a model as valid. 
        residual_threshold (Numeric): Residual (distance) threshold to consider an element as inlier.  
        max_trials (int): Max iterations per line. 
        max_lines (int, optional): Max lines to detect. Defaults to 8.

    Returns:
        List[Tuple[int, int, int, int]]: List of detected lines. A line consists of (x_start, y_start, x_end, y_end)
    """    
    
    # Find coordinates of centroids
    ys, xs = np.where(mask == 255)
    points = np.column_stack([xs, ys])

    lines: List[Tuple[int, int, int, int]] = []
    for _ in range(max_lines):
        if len(points) < min_samples:
            break

        # Apply RANSAC
        ransac = RANSACRegressor(min_samples=min_samples,
                                 residual_threshold=residual_threshold,
                                 max_trials=max_trials)
        ransac.fit(points[:, 0].reshape(-1, 1), points[:, 1])

        # Identify inliers
        inlier_mask = ransac.inlier_mask_
        inliers = points[inlier_mask]

        if len(inliers) < min_samples:
            break
        if len(inliers) < min_inliers:
            break

        # Calculate start and end points using inliers
        x_start, x_end = np.min(inliers[:, 0]), np.max(inliers[:, 0])
        y_start, y_end = ransac.predict([[x_start]])[0], ransac.predict([[x_end]])[0]
        lines.append((int(x_start), int(y_start), int(x_end), int(y_end)))

        # Remove inliers for the next iteration
        points = points[~inlier_mask]

    return lines

