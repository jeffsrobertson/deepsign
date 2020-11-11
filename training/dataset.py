import torch
import os
import imageio
import numpy as np
from scipy import interpolate
import cv2 as cv
from augmentation import *
from utils import generate_codex


class ASLDataset(torch.utils.data.Dataset):
    """
    Wrapper for loading videos from ASL dataset. To call a training video:
        dataset = ASLDataset('/data/ASL', desired_frames=16, desired_size=(112, 112))
        vid, key = dataset[4]
        # vid will be an array of shape (3, 16, 112, 112)
        # key will be an integer, corresponding to the label of the video
        # To get the video's label:
        label = dataset.codex[key]
        # label will be a string (i.e. 'boy'), corresponding to the word signed in the video
    
    Args:
        root_path (string): Root directory of videos.
        desired_frames (int, optional): Number of frames you want to reshape video to be.
        desired_size (tuple, optional): Tuple of shape (height, width), corresponding to number of
            pixels you want the frames of the videos to be.
        trim_longer_side (bool, optional): If true, will crop the longer dim of the video to be same
            size as the shorter side. This is performed before reshaping the frames to be desired_size
            
     Attributes:
        root_path (str): see above
        desired_frames (int): see above
        desired_size (tuple): see above
        trim_longer_side (bool, optional): see above
        codex (dict): Contains mapping of all labels to unique numeric keys.
            i.e. codex = {'boy': 1, 'girl': 2, 'tree': 3, ...}
            Codex is automatically generated from the labels it finds in root_path
        metadata (list): List of dictionaries with information for each training video. Each dict
            contains the following:
                {'path': '/path/to/video.mp4', 'label': 'boy'}
        lexicon (list): List of all unique labels in dataset.
            i.e. lexicon = ['boy', 'tree', 'girl', ...]
    """

    def __init__(self, root_path,
                 desired_frames=16,
                 desired_size=(112, 112),
                 trim_longer_side=True,
                 val_frac=None,
                 val_metadata=None):
        
        self.root_path = root_path
        self.desired_frames = desired_frames
        self.desired_size = desired_size
        self.trim_longer_side = trim_longer_side
        self.val_frac = val_frac
        
        # Creates a list of dicts: [{'path': '/data/ASL/dog/1_handspeak.mp4', 'label': 'dog'}, ...]
        self.metadata = self.get_metadata(root_path)
        
        # If validation dataset, only keep designated fraction of metadata
        if val_frac is not None:
            num_vids_to_keep = int(val_frac*len(self.metadata))
            val_metadata = []
            for i in range(num_vids_to_keep):
                val_metadata.append(np.random.choice(self.metadata))
            self.metadata = val_metadata
            print('>> Reserved {} videos for validation set'.format(num_vids_to_keep))
            
        # If training dataset, remove videos that are in validation dataset
        else:
            if val_metadata is not None:
                print('>> Omitting validation samples from training dataset')
                val_metadata_paths = [item['path'] for item in val_metadata]
                for i in range(len(self.metadata)):
                    vid_path = self.metadata[i]['path']
                    if vid_path in val_metadata_paths:
                        self.metadata[i] = None
                self.metadata = [x for x in self.metadata if x is not None]
            
        
        # Alphebetized list of all unique labels in dataset
        self.lexicon = list(set([item['label'] for item in self.metadata]))
        self.lexicon.sort()
        
        # {'boy': 0, 'dog': 1, 'tree': 2, ...}
        self.codex = generate_codex(self.lexicon)

    def get_metadata(self, root_path):
        """Returns a list of dicts, whose keys/values contain the filepaths 
        of all training vids and their corresponding label, respectively
        
        i.e. [{'/path/to/dog': 'dog'}, {'/path/to/cat': 'cat'}, ...]
        """
        
        metadata_list = []
        for root, subdirs, files in os.walk(root_path):
            for file in files:
                if file.endswith('.mp4'):
                    full_filepath = os.path.join(root, file)
                    label = root.split(sep='/')[-1]
                    metadata_list.append({'path': full_filepath, 'label': label})
        return metadata_list
    
    def load_video(self, path):
        """Loads the video file at location path, and returns a 4D numpy array
        of dim (3, frames, height, width)
        """
        
        try:
            vid = imageio.get_reader(path,  'ffmpeg')
        except:
            raise Exception('Could not load the following video, as it may be corrupt: {}'.format(path))
        
        if not os.path.exists(path):
            path = os.path.join(self.root_path, path)

        list_of_frames = []
        for image in vid.iter_data():
            list_of_frames.append(image)
        
        vid_array = np.array(list_of_frames) # (f, h, w, 3)
        vid_array = np.transpose(vid_array, (3, 0, 1, 2)) # (3, f, h, w)
        
        return vid_array

    def time_transform(self, vid_array, desired_frames, max_crop=.3):
        """Given a 4D video array of dim (3, frames, height, width), 
        compresses/interpolates the video to the number of desired_frames.
        
        Also removes a random amount off either end of the video, from 0 to
        max_crop.
        
        """
        
        _, total_frames, num_rows, num_cols = vid_array.shape
        
        # Crop a random # of frames, and then randomly choose location of crop
        crop_amount = np.random.uniform(low=0, high=max_crop)
        start_crop_amount = np.random.uniform(low=0, high=crop_amount)
        end_crop_amount = crop_amount - start_crop_amount
        start_i = int(start_crop_amount*total_frames)
        end_i = int(total_frames*(1 - end_crop_amount))
        vid_cropped = vid_array[:, start_i:end_i+1, :, :]
        
        # Interp to be 16 frames
        time_axis = np.arange(vid_cropped.shape[1])
        new_time_axis = np.linspace(0, np.max(time_axis), desired_frames)

        # Interpolate vids to all be same number of frames
        new_vid = interpolate.interp1d(time_axis, vid_cropped, axis=1)(new_time_axis)
        
        return new_vid

    def spatial_transform(self, vid_array, min_augments=1, max_augments=2):
        """Applies a random number of augmentations to the vid_array, between
        min_augments and max_augments. The augmentations themselves contain
        their own parameters which are randomized every time they're called."""
        
        
        # Reshape frames
        if self.trim_longer_side:
            height, width = vid_array.shape[2:]
            if width > height:
                vid_array = vid_array[:, :, :, int(height/2): int(width - height/2)]
        reshaped_vid = np.zeros(shape=vid_array.shape[0:2]+self.desired_size)
        for f in range(vid_array.shape[1]):
            for c in range(vid_array.shape[0]):
                reshaped_vid[c, f, :, :] = cv.resize(vid_array[c, f, :, :], dsize=self.desired_size)
        vid_array = reshaped_vid
        
        # Augment
        list_of_transforms = [random_crop, random_horizontal_flip, random_rotate,
                              random_add_intensity, random_multiply_intensity, random_blur]
        num_augments = np.random.randint(min_augments, max_augments+1)
        for n in range(num_augments):
            transform = np.random.choice(list_of_transforms)
            _, vid_array = transform(vid_array)

        return vid_array

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (vid_array, key), where
                vid_array (4d array): array of shape (3, frames, height, width)
                key (int): key corresponding to label. To get the (str) label: self.codex[key]
        """

        # 1. Load vid from path as array (3, f, h, w)path = self.metadata[index]['path']

        path = self.metadata[index]['path']
        label = self.metadata[index]['label']
        vid_array = self.load_video(path)

        # 2. Time transform
        vid_array = self.time_transform(vid_array, self.desired_frames)

        # 3. Apply spatial transforms on list of frames
        vid_array = self.spatial_transform(vid_array)

        key = self.codex[label]

        # clip = tensor of shape (3, f, h, w)
        # label = (int) (i.e. 3)
        return torch.from_numpy(vid_array.copy()), key

    def __len__(self):
        return len(self.metadata)
