import numpy as np
from scipy.ndimage import zoom, rotate
from scipy.ndimage.filters import gaussian_filter

def random_crop(vid_array, min_scale=.7, max_scale=1.3):
    """
    Scales the video frames by some randomly generated value between min_scale and max_scale.
    All frames are scaled by the same scale factor.
    After scaling, randomly picks bounds of new frame, so some translation of the image will occur.

    Input:
        vid_array: (ndarray) 4d array of shape (3, frames, height, width)
        min_scale: (float) Minimum allowed scale factor
        max_scale: (float) Maximum allowed scale factor

    Output:
        scale_factor: (float) Scale factor used for this video
        new_vid_array: (ndarray) 3d array of scaled video, same shape as the input array

    """
    scale_factor = np.random.uniform(low=min_scale, high=max_scale)
    num_colors, num_frames, old_rows, old_cols = vid_array.shape
    new_rows, new_cols = zoom(vid_array[0, 0, :, :], scale_factor).shape

    # If randomly-generated scale is ~1, just return original array
    if new_rows == old_rows:
        return scale_factor, vid_array

    if scale_factor > 1:
        new_x1 = np.random.randint(0, new_cols - old_cols)
        new_x2 = new_x1 + old_cols
        new_y1 = np.random.randint(0, new_rows - old_rows)
        new_y2 = new_y1 + old_rows
    else:
        new_x1 = np.random.randint(0, old_cols - new_cols)
        new_x2 = new_x1 + new_cols
        new_y1 = np.random.randint(0, old_rows - new_rows)
        new_y2 = new_y1 + new_rows

    new_vid_array = np.zeros_like(vid_array)
    for f in range(num_frames):
        new_frame = []
        for c in range(3):
            new_frame.append(zoom(vid_array[c, f, :, :], scale_factor))
        new_frame = np.array(new_frame)

        if scale_factor > 1:
            new_vid_array[:, f, :, :] = new_frame[:, new_y1:new_y2, new_x1:new_x2]

        if scale_factor < 1:
            new_vid_array[:, f, new_y1:new_y2, new_x1:new_x2] = new_frame

    new_vid_array[new_vid_array > 255] = 255
    new_vid_array[new_vid_array < 0] = 0
    return scale_factor, new_vid_array

def random_horizontal_flip(vid_array, flip_chance=.5):
    
    rng = np.random.random()
    flipped = False
    if rng > flip_chance:
        vid_array = np.flip(vid_array, axis=-1)
        flipped = True
        
    vid_array[vid_array > 255] = 255
    vid_array[vid_array < 0] = 0
    return flipped, vid_array

def random_rotate(vid_array, min_degrees=-8, max_degrees=8):
    """
    Rotates the video frames  by some randomly generated value between min_degrees and and max_degrees.
    All frames are rotated by the same degree.

    Input:
        vid_array: (ndarray) 4d array of shape (3, frames, height, width)
        min_degrees: (float) minimum allowed degree to rotate
        max_degrees: (float) maximum allowed degree to rotate

    Output:
        degree: (float) degree used to rotate this video
        new_vid_array: (ndarray) 4d array of rotated video, same shape as input array

    """
    degree = np.random.uniform(low=min_degrees, high=max_degrees)

    new_vid_array = rotate(vid_array, degree, reshape=False, axes=(2, 3))

    new_vid_array[new_vid_array > 255] = 255
    new_vid_array[new_vid_array < 0] = 0
    return degree, new_vid_array


def random_multiply_intensity(vid_array, min_scale=.9, max_scale=1.1):
    """
    Uniformly multiplies the video intensity by a randomly chosen value between min_scale and max_scale.
    Pixel values are automatically capped at 1.

    Input:
        vid_array: (ndarray) 4d array of shape (3, frames, height, width)
        max_scale: (float) maximum allowed multiplicative factor for image intensity.
        min_scale: (float) minimum allowed multiplicative factor for image intensity.

    Output:
        scale_factor: (float) scale factor used in generating new video.
        new_vid_array: (ndarray) 4d array, same shape as input array

    """

    if min_scale < 0:
        raise ValueError("min_noise parameter for salt_and_pepper() must be greater than 0.")
    if min_scale > max_scale:
        raise ValueError("max_scale must be greater than min_scale in multiply_intensity()")

    scale_factor = np.random.uniform(min_scale, max_scale)

    new_vid_array = scale_factor*vid_array

    new_vid_array[new_vid_array > 255] = 255
    new_vid_array[new_vid_array < 0] = 0
    return scale_factor, new_vid_array


def random_add_intensity(vid_array, min_add=-.3, max_add=.3):
    """
    Uniformly adds a value to all pixel intensities.
    Additive value is randomly selected to be between min_add*np.max(vid_array) and max_add*np.max(vid_array)
    Pixel values are automatically capped to be between 0 and 1.

    Input:
        vid_array: (ndarray) 4d array of shape (color, frames, height, width)
        max_add: (float) maximum allowed additive factor for image intensity.
        min_scale: (float) minimum allowed additive factor for image intensity.

    Output:
        add_factor: (float) Additive factor used in generating new video.
        new_vid_array: (ndarray) 4d array of modified video, same shape as input array.

    """
    
    if min_add > max_add:
        raise ValueError("max_add must be greater than min_add in random_add_intensity()")

    add_factor = np.random.uniform(min_add, max_add)

    new_vid_array = vid_array + add_factor*np.max(vid_array)

    new_vid_array[new_vid_array > 255] = 255
    new_vid_array[new_vid_array < 0] = 0
    return add_factor, new_vid_array

def random_blur(vid_array, min_sigma=0, max_sigma=.01):
    """
    Applies a gaussian blur to the image.
    Standard deviation of blur is randomly choseen between min_sigma and max_sigma.
    All frames/color channels are blurred by the same amount.

    Input:
        vid_array: (ndarray) 4d array of shape (color, frames, height, width)
        max_sigma: (float) maximum allowed stdev of gaussian.
        min_sigma: (float) minimum allowed stdev of gaussian.

    Output:
        add_factor: (float) Additive factor used in generating new video.
        new_vid_array: (ndarray) 3d array of modified video, same shape as input array.

    """
    num_colors, num_frames, num_rows, num_cols = vid_array.shape
    sigma_factor = np.random.uniform(min_sigma, max_sigma)
    sigma = num_rows*sigma_factor
    
    blurred_vid = np.zeros_like(vid_array)
    for f in range(num_frames):
        for c in range(num_colors):
                blurred_vid[c, f, :, :] = gaussian_filter(vid_array[c, f, :, :], sigma)
                
    blurred_vid[blurred_vid > 255] = 255
    blurred_vid[blurred_vid < 0] = 0
    return sigma, blurred_vid



