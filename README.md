# DeepSign

DeepSign is an interactive app for teaching users how to speak American Sign Language (ASL).

![demo](media/demo.gif)

The app gives the user a random word to sign, and then accesses the user's webcam in order to analyze the video footage (via a 3D ConvNet) and give feedback on if the word was signed correctly.

While many neural networks exist for recognizing still images of the ASL alphabet (a problem that has been beaten to death on Kaggle), this is the first app of its kind which is able to recognize animated animated gestures for signed words (at least, to my knowledge). This app serves as a proof-of-concept of such an app and, despite the severe shortage of publicly-available data to train on, is largely successful in achieving accurate gesture recognition across several dozen unique words.

This app was a project that I built as part of the [Insight](https://insightfellows.com/data-science) data science fellowship. I recorded a video of me rambling about it [here](https://www.youtube.com/watch?v=7WUowREyM6o), if you'd prefer to listen to my beautiful voice.

## To run this app

This app is run locally in the user's web browser. To use the app:

1. Download and extract the repository to your computer
2. In the terminal, go to the directory containing main.py
3. Install necessary dependencies (see below)
4. Run the command "python main.py"
5. The terminal will print a bunch of lines, specifically containing one that looks like the following:

> Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

Paste this url into your web browser and it should load the app.

### Dependencies

This app uses OpenCV, Pytorch, and Flask, which can be installed via the following pip commands:

> pip install opencv-python
>
> pip install torch
>
> pip install Flask

Alternatively, a requirements.txt file has been provided, and so you can create a virtual env using the following:

> python -m venv venv
>
> source venv/bin/activate
>
> pip install -r requirements.txt

This app was tested in python 3.8.3, it should work for all python 3+ versions.

### Tips

- When you open the app, you should get a popup asking for permission to access your webcam. If not, you may need to go into your browser's settings and enable webcam access for the site 127.0.0.1
- For best results, try to align your body with the faint outline of the body that's drawn on the video feed (it doesn't have to be perfect, but it helps)
- The grayed out areas on either side of the video are discarded by the neural network

## Training

The folder "training/" contains all the notebooks and scripts I used to collect training data and train the neural network. This folder is not needed to run the app. Some details about the files within:

- **training/web_scraper.ipynb** - Scrapes the web for training videos, using BeautifulSoup for html parsing.
- **training/trainer.ipynb** - Loads a pretrained neural network (trained on the Jester dataset - see below), then applies transfer learning to fine-tune the last couple layers of the network on the ASL videos that were gathered in step (1).


## About this app

Gesture recognition is achieved through a 3D convolutional neural network. In a 3D ConvNet, a sequence of frames are fed into the neural network as an input "volume" with dimension of height, width, and time. A 3D kernel is then convolved across the input volume in order to extract spatiotemporal features from the video. The neural network architecture that was used is based on a 3D generalization of Google's MobileNetV2 network, so that it can be run efficiently on devices which only have CPU capabilities.

Because no open source dataset exists for animated ASL gestures (at least that I'm aware of), one of the biggest challenges in developing this app was acquiring enough data to effectively train a neural network. As is common in many computer vision applications, the shortage of data was addressed by using a pretrained model, which was trained on the [Jester dataset](https://20bn.com/datasets/jester).

![Jester](media/jester.gif)

The Jester dataset consists of ~150,000 videos of people performing hand gestures across 27 different categories, and is one of the best datasets available for training gesture-recognition neural networks. If you go to the "Gesture" tab in the app, you can play with a NN that was trained on the Jester dataset, and test out the various hand gestures.

After training a NN from scratch on Jester, the last couple layers were replaced and retrained on a much smaller dataset of ASL hand gestures. ASL training videos were acquired by scraping the web using the python library BeautifulSoup (you can find the code under web_scraper.ipynb), for as many "ASL dictionary" sites I could find - which amounted to a whopping ~15 unique videos per word (compare that to the ~5k+ videos/gesture from Jester). This was, surprisingly, enough videos to yield reasonably accurate results, although the app does sometimes throw up a false positive. I experimented a *lot* with the hyperparameters of this NN, and it does pretty well on classifying up to ~30 words simultaneously, but after that it begins to struggle due to the lack of data.

The neural networks discussed above were all trained remotely on an AWS EC2 instance, using 8 GPUs in parallel. While I did a lot of prototyping in Keras/Tensorflow, I ultimately used Pytorch to train my neural network, as it allows a lot more customization and its API makes it super easy to leverage CUDA-powered parallel GPU computing.


## About me

I'm a Physics PhD-turned-data-scientist, with an emphasis on computer vision and deep learning methods. If you have any questions about me, this project, or something else, feel free to reach out:

* [linkedin](https://www.linkedin.com/in/jeffsrobertson/)

## Future releases

The app currently must be run locally within the user's web browser. I am working on a live web app version, so that anyone can use the app simply by going to a url, but this requires re-coding a good chunk of the app to allow for server-client communication.

## License

There is none. Download it, use it, sell it, go nuts. This app was built as a passion project to see if I could develop an effective classification system in an area which remains relatively unexplored.


