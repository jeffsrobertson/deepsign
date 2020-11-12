import threading
import time
import numpy as np
from .model.mobilenetv2 import MobileNetV2
import torch
from scipy import interpolate
from collections import OrderedDict
from torch.autograd import Variable
import os
import cv2 as cv


class ASLWebCam:

    def __init__(self):

        self.cap = None

        self.codex = {'again': 1,
                      'alligator': 2, 'bird': 3, 'book': 4,
                      'boy': 5, 'but': 6, 'cat': 7, 'family': 8, 'girl': 9,
                      'grandmother': 10, 'happy': 11, 'house': 12,
                      'no_gesture': 13, 'please': 14, 'tired': 15}
        self.inv_codex = {v: k for k, v in self.codex.items()}
        self.model_path = 'app/model/asl_model.pth'

        self.current_frame = None

        # Rate at which you save screenshots to memory
        self.sample_rate = 12  # [Hz]
        self.max_sampled_frames = 36
        self.sampled_times = []
        self.last_sampled_time = 0
        self.sampled_frames = []

        #  Parameters for the saved array that goes through the NN
        self.prediction_rate = 2  # [Hz]
        self.prediction_duration = 2  # [s]
        self.last_predicted_time = time.time()
        self.predicted_word = ''
        self.predicted_word_prob = 0
        self.num_active_predictions = 0
        self.max_num_predictions = 1

        self.fps_list = [0]*25

        self.turned_on = False

        self.model = self.load_model()

        self.word = 'no_gesture'
        self.outline_img = cv.imread('app/static/outline.png')

        self.next_guess_time = 0
        self.guess_message = ''
        self.guess_color = (255, 255, 255)
        self.already_guessed_words = []
        self.total_correct_tally = 0
        self.allow_tally_reward = True

        self._base_fontscale = 1.

        print('>> Initialized webcam')

    def _capture_loop(self):

        print('>> Turning on main camera loop')
        while self.turned_on:
            current_time = time.time()
            ret, frame = self.cap.read()

            if not ret:
                continue

            # Sample and save frames that will go through NN

            if current_time > self.last_sampled_time + (1/self.sample_rate):
                self.sample_frame(frame, current_time)

            # if current_time > self.last
            if current_time > self.last_predicted_time + (1/self.prediction_rate):
                if len(self.sampled_frames) == self.max_sampled_frames:
                    if self.num_active_predictions < self.max_num_predictions:
                        thread = threading.Thread(target=self.predict_on_model)
                        thread.daemon = True
                        thread.start()
                        self.last_predicted_time = time.time()

            # Draw frame
            self.current_frame = self.draw_frame(frame)

            self.calc_fps(current_time)

        self.cap.release()
        print('>> Safely ended camera thread.')

    def predict_on_model(self):

        # Let main loop know it's busy calculating the model prediction
        self.num_active_predictions += 1

        # Find closest index corresponding to 2 seconds ago
        key_time = self.sampled_times[-1] - self.prediction_duration
        last_index = np.argmin(np.abs(np.array(self.sampled_times) - key_time))
        desired_frames = self.sampled_frames[last_index:]
        desired_times = self.sampled_times[last_index:]

        time_axis_old = np.array(desired_times)
        time_axis_new = np.linspace(np.min(time_axis_old), np.max(time_axis_old), 16)

        # Each frame is shape (h, w, 3)
        vid_array = np.stack(desired_frames, axis=0)
        vid_array = np.transpose(vid_array, (3, 0, 1, 2))

        # vid_array is shape (3, f, h, w)
        vid_array = interpolate.interp1d(time_axis_old, vid_array, axis=1)(time_axis_new)
        vid_array = np.expand_dims(vid_array, axis=0)

        # Determine predicted word and associated probability
        vid_array = torch.FloatTensor(vid_array.copy())
        output = self.model(vid_array)
        probabilities = torch.nn.functional.softmax(Variable(output), dim=1).data
        probabilities = np.squeeze(probabilities.numpy())
        max_i = int(np.argmax(probabilities))
        self.predicted_word_prob = 100*probabilities[max_i]#'{:.1f}'.format(100*probabilities[max_i])+'%'
        self.predicted_word = self.inv_codex[max_i + 1]

        # Let main loop know it's done calculating prediction
        self.num_active_predictions -= 1


    def load_model(self):
        print('>> Loading pytorch neural network.')
        model = MobileNetV2(num_classes=len(self.codex), sample_size=112, width_mult=1.)
        if not torch.cuda.is_available():
            print('>> No CUDA detected, loading model onto CPU')
            full_filepath = os.path.join(os.getcwd(), self.model_path)
            loaded_state_dict = torch.load(full_filepath, map_location=torch.device('cpu'))['state_dict']
            state_dict = OrderedDict()
            for k, v in loaded_state_dict.items():
                if k == 'module.classifier.weight':
                    state_dict["classifier.1.weight"] = v
                elif k == 'module.classifier.bias':
                    state_dict['classifier.1.bias'] = v
                else:
                    state_dict[k[7:]] = v
        else:
            print('>> CUDA detected, running neural network on GPU')
            state_dict = torch.load(self.model_path)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    def sample_frame(self, frame, current_time):

        # Center crop (height x height x 3) square
        height, width, _ = frame.shape
        x1 = int(.5*(width-height))
        center_cropped_frame = frame[:, x1:x1+height, :]

        # Shrink to 112 x 112 x 3
        sample = cv.cvtColor(src=center_cropped_frame, code=cv.COLOR_BGR2RGB)
        sample = cv.resize(src=sample, dsize=(112, 112))

        # Add to list of cached frames
        self.sampled_frames.append(sample)
        self.sampled_times.append(current_time)
        if len(self.sampled_frames) > self.max_sampled_frames:
            self.sampled_frames.pop(0)
            self.sampled_times.pop(0)
        self.last_sampled_time = current_time

    def draw_frame(self, frame):

        # Draw frame
        height,  width, _ = frame.shape

        # Add body outline, grey out region of frame outside it
        frame = cv.addWeighted(src1=frame, alpha=1., src2=self.outline_img, beta=0.1, gamma=0)
        gray_frame = cv.cvtColor(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
        x1 = int(.5*(width - height))
        frame[:, 0:x1, :] = gray_frame[:, 0:x1, :]
        frame[:, width - x1:, :] = gray_frame[:, width-x1:, :]

        # flip horizontally (so image is mirror of user)
        frame = cv.flip(frame, flipCode=1)

        # Add fps to bottom right corner
        message = 'fps: '+'{:.2f}'.format(np.mean(self.fps_list))
        (w, h), _ = cv.getTextSize(message, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=self._base_fontscale, thickness=2)
        cv.putText(img=frame,
                   text=message,
                   org=(int(.99*width - w), int(.99*height)),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                   fontScale=self._base_fontscale, 
                   color=(0, 0, 0),
                   thickness=2)

        # Give time for initial greeting to display
        if len(self.sampled_frames) < int(.5*self.max_sampled_frames):
            message = 'Welcome to DeepSign!'
        else:
            message = 'Sign the following word: '
        cv.putText(img=frame, 
                   text=message, 
                   org=(int(.01*width), int(.07*height)),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1.4*self._base_fontscale, 
                   color=(255, 153, 0), 
                   thickness=3)

        # Only print out word if the list of sample frames has saturated
        if len(self.sampled_frames) == self.max_sampled_frames:
            cv.putText(img=frame,
                       text=self.word,
                       org=(int(.1*width), int(5*self._base_fontscale*22)),
                       fontFace=cv.FONT_HERSHEY_COMPLEX,
                       fontScale=1.8*self._base_fontscale,
                       color=(255, 153, 0),
                       thickness=5)

        # Say whether guess is correct/incorrect
        if time.time() > self.next_guess_time:
            if not self.predicted_word == 'no_gesture':
                if self.predicted_word_prob > 50:
                    if self.predicted_word == self.word:
                        self.guess_color = (51, 204, 51)
                        self.guess_message = 'Good job!'
                        self.next_guess_time = time.time() + 3

                        # ++ correct word tally
                        if self.allow_tally_reward:
                            self.total_correct_tally += 1
                        self.allow_tally_reward = False
                    else:
                        self.guess_color = (0, 0, 255)
                        self.guess_message = 'Try again'
                        self.next_guess_time = time.time() + 1
                else:
                    self.guess_message = ''
                    self.guess_color = (255, 255, 255)
            else:
                self.guess_message = ''
                self.guess_color = (255, 255, 255)

        # Print out most recent prediction from NN
        if len(self.sampled_frames) == self.max_sampled_frames:
            cv.putText(img=frame,
                       text=self.guess_message,
                       org=(int(.025*width), int(.25*height+3*self._base_fontscale*22)),
                       fontFace=cv.FONT_HERSHEY_SCRIPT_COMPLEX, 
                       fontScale=3*self._base_fontscale, 
                       color=self.guess_color,
                       thickness=5)
            cv.putText(img=frame, 
                       text=self.predicted_word+'  '+str(round(self.predicted_word_prob, 2))+'%', 
                       org=(int(.01*width), int(.99*height)),
                       fontFace=cv.FONT_HERSHEY_COMPLEX, 
                       fontScale=1.4*self._base_fontscale, 
                       color=self.guess_color,
                       thickness=2)

        # Correct tally
        message = 'Total correct: {}'.format(self.total_correct_tally)
        (w, h), _ = cv.getTextSize(message, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.4*self._base_fontscale, thickness=3)
        cv.putText(img=frame,
                   text=message,
                   org=(int(.97*width - w), int(.03*height+h)),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1.4*self._base_fontscale,
                   color=(255, 153, 0),
                   thickness=3)

        return frame

    def get_frame(self):

        if self.current_frame is not None:
            img = cv.imencode('.png', self.current_frame)[1].tobytes()
        else:
            with open("app/static/not_found.jpeg", "rb") as f:
                img = f.read()
        return img

    def update_guess_word(self):

        self.already_guessed_words.append(self.word)
        list_of_words = list(self.codex.keys())
        list_of_words = [item for item in list_of_words if ((item != 'no_gesture') and (item not in self.already_guessed_words))]
        if len(list_of_words) == 0:
            print('>> Resetting available word list')
            self.already_guessed_words = []
            self.update_guess_word()
            return
        new_word = np.random.choice(list_of_words)
        
        self.word = new_word
        self.allow_tally_reward = True
        print('>> New word: {}'.format(self.word))

    # Calculates the moving-average fps
    def calc_fps(self, current_time):
        end_time = time.time()
        latest_fps = 1.0/(end_time - current_time)
        self.fps_list.append(latest_fps)
        self.fps_list.pop(0)

    def turn_on(self):
        self.cap = cv.VideoCapture(0)
        self.turned_on = True

        # Resize body outline img to match user's webcam output
        cap_width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        cap_height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        if self.outline_img.shape[0:2] != (cap_height, cap_width):
            self.resize_outline((cap_height, cap_width))

        # Initialize all fonts to be proportional to image height
        self._base_fontscale = cap_height/720.

        # Start camera loop on new thread
        thread = threading.Thread(target=self._capture_loop)
        thread.daemon = True
        thread.start()

    def turn_off(self):
        if self.cap is not None:
            self.turned_on = False

            # Freeze up main thread until the webcam has had time to close
            while self.cap.isOpened():
                continue
            self.already_guessed_words = []
            self.sampled_frames = []
            self.sampled_times = []
            self.guess_message = ''
            self.predicted_word = ''
            self.total_correct_tally = 0
            self.allow_tally_reward = True

    def resize_outline(self, desired_shape):

        cap_height, cap_width = desired_shape
        outline_height, outline_width, _ = self.outline_img.shape
        scale_factor = cap_height/outline_height
        new_outline_height = int(cap_height)
        new_outline_width = int(scale_factor*outline_width)
        scaled_outline = cv.resize(self.outline_img, (new_outline_width, new_outline_height))
        new_outline = np.full((int(cap_height), int(cap_width), 3), 255, dtype=np.uint8)
        x1 = int(.5*(cap_width - new_outline_width))
        new_outline[:, x1:x1+new_outline_width, :] = scaled_outline
        self.outline_img = new_outline


class JesterWebCam:

    def __init__(self):

        # Load desired model into memory
        self.cap = None
        self.inv_codex = {1: '',
                          2: 'drumming fingers',
                          3: 'no gesture',
                          4: 'pulling hand in',
                          5: 'pulling two fingers in',
                          6: 'pushing hand away',
                          7: 'pushing two fingers away',
                          8: 'rolling hand backward',
                          9: 'rolling hand forward',
                          10: 'shaking hand',
                          11: 'sliding two fingers down',
                          12: 'sliding two fingers left',
                          13: 'sliding two fingers right',
                          14: 'sliding two fingers up',
                          15: 'stop sign',
                          16: 'swiping down',
                          17: 'swiping left',
                          18: 'swiping right',
                          19: 'swiping up',
                          20: 'thumb down',
                          21: 'thumb up',
                          22: 'turning hand clockwise',
                          23: 'turning hand counterclockwise',
                          24: 'zooming in with full hand',
                          25: 'zooming in with two fingers',
                          26: 'zooming out with full hand',
                          27: 'zooming out with two fingers'}
        self.codex = {v: k for k, v in self.inv_codex.items()}

        self.current_frame = None

        # Rate at which you save screenshots to memory
        self.sample_rate = 12  # [Hz]
        self.max_sampled_frames = 36
        self.sampled_times = []
        self.last_sampled_time = 0
        self.sampled_frames = []

        #  Parameters for the saved array that goes through the NN
        self.prediction_rate = 2  # [Hz]
        self.prediction_duration = 2  # [s]
        self.last_predicted_time = time.time()
        self.predicted_word = ''
        self.predicted_word_prob = 0
        self.num_active_predictions = 0
        self.max_num_predictions = 1

        self.fps_list = [0]*25

        self.turned_on = False

        self.model = self.load_model()

        self._base_fontscale = 1.

        print('>> Initialized webcam')

    def _capture_loop(self):

        print('>> Turning on main camera loop')
        while self.turned_on:
            current_time = time.time()
            ret, frame = self.cap.read()

            if not ret:
                continue

            # Sample and save frames that will go through NN
            if current_time > self.last_sampled_time + (1/self.sample_rate):
                self.sample_frame(frame, current_time)

            # if current_time > self.last
            if current_time > self.last_predicted_time + (1/self.prediction_rate):
                if len(self.sampled_frames) == self.max_sampled_frames:
                    if self.num_active_predictions < self.max_num_predictions:
                        thread = threading.Thread(target=self.predict_on_model)
                        thread.daemon = True
                        thread.start()
                        self.last_predicted_time = time.time()

            # Draw frame
            self.current_frame = self.draw_frame(frame)

            self.calc_fps(current_time)

        self.cap.release()
        print('>> Safely ended camera thread.')

    def sample_frame(self, frame, current_time):

        # Center crop (height x height x 3) square
        height, width, _ = frame.shape
        x1 = int(.5*(width-height))
        center_cropped_frame = frame[:, x1:x1+height, :]

        # Shrink to 112 x 112 x 3
        sample = cv.cvtColor(src=center_cropped_frame, code=cv.COLOR_BGR2RGB)
        sample = cv.resize(src=sample, dsize=(112, 112))

        # Add to list of cached frames
        self.sampled_frames.append(sample)
        self.sampled_times.append(current_time)
        if len(self.sampled_frames) > self.max_sampled_frames:
            self.sampled_frames.pop(0)
            self.sampled_times.pop(0)
        self.last_sampled_time = current_time

    def predict_on_model(self):

        # Let main loop know it's busy calculating the model prediction
        self.num_active_predictions += 1

        # Find closest index corresponding to 2 seconds ago
        key_time = self.sampled_times[-1] - self.prediction_duration
        last_index = np.argmin(np.abs(np.array(self.sampled_times) - key_time))
        desired_frames = self.sampled_frames[last_index:]
        desired_times = self.sampled_times[last_index:]

        time_axis_old = np.array(desired_times)
        time_axis_new = np.linspace(np.min(time_axis_old), np.max(time_axis_old), 16)

        # Each frame is shape (h, w, 3)
        vid_array = np.stack(desired_frames, axis=0)
        vid_array = np.transpose(vid_array, (3, 0, 1, 2))

        # vid_array is shape (3, f, h, w)
        vid_array = interpolate.interp1d(time_axis_old, vid_array, axis=1)(time_axis_new)
        vid_array = np.expand_dims(vid_array, axis=0)

        # Determine predicted word and associated probability
        vid_array = torch.FloatTensor(vid_array.copy())
        output = self.model(vid_array)
        probabilities = torch.nn.functional.softmax(Variable(output), dim=1).data
        probabilities = np.squeeze(probabilities.numpy())
        max_i = int(np.argmax(probabilities))
        self.predicted_word_prob = 100*probabilities[max_i]
        self.predicted_word = self.inv_codex[max_i + 1]

        # Let main loop know it's done calculating prediction
        self.num_active_predictions -= 1

    def draw_frame(self, frame):
        
        # Draw frame
        height, width, _ = frame.shape
        
        # Flip horizontally
        frame = cv.flip(frame, flipCode=1)
        
        # Add fps to bottom right
        message = 'fps: '+'{:.2f}'.format(np.mean(self.fps_list))
        (w, h), _ = cv.getTextSize(message, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=self._base_fontscale, thickness=2)
        cv.putText(img=frame, 
                   text=message, 
                   org=(int(.99*width - w), int(.99*height)),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                   fontScale=self._base_fontscale, 
                   color=(0, 0, 0),
                   thickness=2)

        # If gesture is none or null, don't print probability
        if self.predicted_word == '' or self.predicted_word == 'no gesture':
            prob_printout = ''
        else:
            prob_printout = self.predicted_word+' ('+str(round(self.predicted_word_prob))+'%)'
        
        # Add last gesture prediction
        cv.putText(img=frame, 
                   text='Last gesture: '+prob_printout,
                   org=(int(.01*width), int(.99*height)), 
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1.5*self._base_fontscale, 
                   color=(255, 0, 255), 
                   thickness=3)

        return frame

    def turn_on(self):
        
        self.cap = cv.VideoCapture(0)
        self.turned_on = True
        
        # Initialize font sizes to be proportional to height of video
        cap_height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        self._base_fontscale = cap_height/720.

        # Start camera loop on new thread
        thread = threading.Thread(target=self._capture_loop)
        thread.daemon = True
        thread.start()

    def turn_off(self):
        if self.cap is not None:
            self.turned_on = False

            # Freeze up main thread until the webcam has had time to close
            while self.cap.isOpened():
                continue
            
            self.__init__()

    def get_frame(self):

        if self.current_frame is not None:
            img = cv.imencode('.png', self.current_frame)[1].tobytes()
        else:
            with open("app/static/not_found.jpeg", "rb") as f:
                img = f.read()
        return img

    # Calculates the moving-average fps
    def calc_fps(self, current_time):
        end_time = time.time()
        latest_fps = 1.0/(end_time - current_time)
        self.fps_list.append(latest_fps)
        self.fps_list.pop(0)

    def load_model(self):
        print('>> Loading pytorch neural network.')
        model = MobileNetV2(num_classes=len(self.codex), sample_size=112, width_mult=1.)
        if not torch.cuda.is_available():
            full_filepath = os.path.join(os.getcwd(), 'app/model/gesture_model.pth')
            loaded_state_dict = torch.load(full_filepath, map_location=torch.device('cpu'))
            state_dict = OrderedDict()
            for k, v in loaded_state_dict.items():
                state_dict[k[7:]] = v
        else:
            print('>> CUDA detected, running neural network on GPU')
            state_dict = torch.load('app/model/gesture_model.pth')
        model.load_state_dict(state_dict)
        model.eval()

        return model









