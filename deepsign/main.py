from flask import Flask, render_template, url_for, Response
from flask import request, redirect
from deepsign.camera import ASLWebCam, JesterWebCam

app = Flask(__name__)
asl_cam = ASLWebCam()
jester_cam = JesterWebCam()

VID_URLS = {'again': "https://www.youtube.com/embed/wTNv94AY-yE",
            'alligator': "https://www.youtube.com/embed/fe8_PKtzrvU",
            'bird': "https://www.youtube.com/embed/Bibgy-yjgYE",
            'book': "https://www.youtube.com/embed/XjWSfh50kAU",
            'boy': "https://www.youtube.com/embed/5H6OSAy-Mzs",
            'but': "https://www.youtube.com/embed/l3pElzy2Png",
            'cat': "https://www.youtube.com/embed/ekFrFoJ-x78",
            'family': "https://www.youtube.com/embed/SxrxUgeTt00",
            'girl': "https://www.youtube.com/embed/pwh3cOdoiG4",
            'grandmother': "https://www.youtube.com/embed/r8yvyGMnrYk",
            'happy': "https://www.youtube.com/embed/SHc7_8aD9Rw",
            'house': "https://www.youtube.com/embed/lBSZYk72vmY",
            'please': "https://www.youtube.com/embed/rnb9FxPO7is",
            'tired': "https://www.youtube.com/embed/YS5PgjNNxME"}


@app.route("/")
@app.route("/asl", methods=['GET', 'POST'])
def asl():
    asl_cam.turn_off()
    jester_cam.turn_off()
    return render_template('asl.html', title='ASL Recognition')


@app.route("/asl_webcam", methods=['GET', 'POST'])
def asl_webcam():
    if request.method == 'GET':
        return redirect(url_for('asl'))
    else:
        asl_cam.turn_on()
        asl_cam.update_guess_word()
        vid_url = VID_URLS[asl_cam.word]
        return render_template('asl_webcam.html',
                               title='ASL Recognition', vid_url=vid_url)


@app.route('/generate_new_word')
def generate_new_word():
    asl_cam.update_guess_word()
    return VID_URLS[asl_cam.word]


@app.route("/gestures", methods=['GET', 'POST'])
def gestures():
    asl_cam.turn_off()
    jester_cam.turn_off()
    return render_template('gestures.html', title='Gesture Recognition')


@app.route("/gestures_webcam", methods=['GET', 'POST'])
def gestures_webcam():
    if request.method == 'GET':
        return redirect(url_for('gestures'))
    else:
        jester_cam.turn_on()
        lexicon = list(set(jester_cam.codex.keys()))
        lexicon.sort()
        lexicon = [word for word in lexicon if word not in ('no gesture', '')]
        return render_template('gestures_webcam.html',
                               title='Gesture Recognition',
                               lexicon=lexicon)


@app.route("/about")
def about():
    return render_template('about.html', title='About Me', about_page=True)


def gen(cam):
    while cam.turned_on:
        frame = cam.get_frame()
        a = (b'--frame\r\n'
             b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
        yield a


@app.route("/video_feed_asl")
def video_feed_asl():
    return Response(gen(asl_cam),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_gesture")
def video_feed_gesture():
    return Response(gen(jester_cam),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
