import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import requests


def download_video(url, save_path):
    """
    Downloads video from url to save_path.

    Input:
        url: (str) full url of mp4 video to download
        save_path: (str) full path of desired save locatiion
    """

    # download video
    r = requests.get(url, stream = True)
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size = 1024*1024):
            if chunk:
                f.write(chunk)


def adjust_learning_rate(new_lr, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 'decay_rate' epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def generate_codex(lexicon, inverse_codex=False):
    """
    Given a list of words, creates a codex of unique numeric keys assigned to the words.

    Example:
        input: ['boy', 'apple', 'dog', 'cat']
        output: {'apple': 0, 'boy': 1, 'cat': 2, 'dog': 3}

    Input:
        lexicon: (list) List of strings, corresponding to all possible labels (words) in training set
        inverse_codex: (Bool) Set to true to flip the order of keys and values in codex

    Output:
        codex: (dict) Dictionary of unique numeric keys for every word in lexicon (see above).

    """

    lexicon.sort()

    codex = {}
    for i in range(len(lexicon)):
        word = lexicon[i]
        if inverse_codex:
            codex[i] = word
        else:
            codex[word] = i

    return codex


def animate_movie(vid_array):
    """
    Takes a 3d array and returns an animation of it.

    Input:
        vid_array: (ndarray) 4d array of shape (3, frames, rows, cols)

    Output:
        anim: (FuncAnimation) matplotlib animation object

    """
    
    fig, ax = plt.subplots()

    # Create initial plot
    frame = np.transpose(vid_array[:, 0, :, :]/255, (1, 2, 0))
    ax.imshow(frame)
    plt.title('Frame {}'.format(0))

    def update(t):
        ax.clear()
        frame = np.transpose(vid_array[:, t, :, :]/255, (1, 2, 0))
        ax.imshow(frame)
        plt.title('Frame {}'.format(t))

        return ax,

    anim = animation.FuncAnimation(fig, update, frames=vid_array.shape[1], interval=1000/10, blit=True)

    return anim

def vid_to_array(filepath):
    """
    Converts a video file to a 3dim greyscale array.

    Input:
        filepath: (str) full filepath of video

    Output:
        vid: (ndarray) numpy array of shape (color, frames, height, width)
    """

    # returns list of (h, w, 3) color images
    list_of_frames = vid_to_list(filepath)

    num_frames = len(list_of_frames)
    if num_frames == 0:
        return np.zeros(shape=(0, 0, 0, 0))
    height, width, color = list_of_frames[0].shape

    vid = np.zeros(shape=(color, num_frames, height, width))
    for i in range(num_frames):
        frame = list_of_frames[i]
        frame = np.transpose(frame, (2, 0, 1)) # change to (3, h, w)
        vid[:, i, :, :] = frame

    return vid

def vid_to_list(filepath):
    """
    Converts a video file to a list of 3d arrays of dim (h, w, c)

    Input:
        filepath: (str) full filepath of video

    Output:
        vid: (ndarray) list of 3d numpy arrays, of shape (height, width, color)
    """

    cap = cv.VideoCapture(filepath)

    list_of_frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            list_of_frames.append(frame)
        else:
            break

    return list_of_frames

def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # pred = index in each batch corresponding to max value
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res
