import torch
import torch.nn.functional as tnf
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from torchvision import transforms
import json
from moviepy.video.fx import resize, loop
import time


class MakeMask:
    def __init__(self, cuda):
        self.cuda = cuda
        self.model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
        self.people_class = 15

        self.model.eval()
        print("Model Loaded")

        self.blur = torch.FloatTensor([[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]]) / 16.0

        # move the input and model to GPU for speed if available ?
        if self.cuda and torch.cuda.is_available():
            print("Using GPU (CUDA) to process the images")
            self.model.to('cuda')
            self.blur = self.blur.to('cuda')

        self.preprocess = transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    def __call__(self, img):
        frame_data = torch.FloatTensor(img) / 255.0

        input_tensor = self.preprocess(frame_data.permute(2, 0, 1))
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available ?
        if self.cuda and torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]

        segmentation = output.argmax(0)

        bgout = output[0:1][:][:]
        a = (1.0 - tnf.relu(torch.tanh(bgout * 0.30 - 1.0))).pow(0.5) * 2.0

        people = segmentation.eq(torch.ones_like(segmentation).long().fill_(self.people_class)).float()

        people.unsqueeze_(0).unsqueeze_(0)

        for i in range(3):
            people = tnf.conv2d(people, self.blur, stride=1, padding=1)

        # combined_mask = tnf.hardtanh(a * b)
        combined_mask = tnf.relu(tnf.hardtanh(a * (people.squeeze().pow(1.5))))
        combined_mask = combined_mask.expand(1, 3, -1, -1)

        newimg = (combined_mask * 255.0).cpu().squeeze().byte().permute(1, 2, 0).numpy()

        return newimg


def processmovie(config):
    inp = config["input"]
    scaler = config["scaler"]

    print(f"Starting to remove the background of {inp}")
    usedclip = VideoFileClip(inp, resize_algorithm=scaler)

    # if won't output the mask, reserve the orignal file to mix after
    bg = config["background"]
    if bg != "":
        originalclip = usedclip.copy()

    rres = config["relative_mask_resolution"] * 0.01
    if rres != 1:
        oldx, oldy = usedclip.size
        newx = oldx * rres
        newy = oldy * rres
        print(f"Decreasing mask resolution in {(1 - rres) * 100}%. {newx}x{newy} now")
        usedclip = usedclip.fx(resize, rres, method=scaler)
    rfps = config["relative_mask_fps"] * 0.01
    if rfps != 1:
        newfps = usedclip.fps * rfps
        print(f"Decreasing mask fps in {(1 - rfps) * 100}%. {newfps}fps now")
        usedclip = usedclip.set_fps(newfps)

    # Get the mask in video format
    maskoutput = usedclip.fl_image(MakeMask(config["cuda"]))
    #usedclip.close()

    if bg != "":
        maskedclip = originalclip.set_mask(maskoutput.to_mask())
        #originalclip.close()
        #maskoutput.close()

        if type(bg) == list:
            rgb = (bg[0], bg[1], bg[2])
            print(f"Using the RGB color {rgb} as the background of {inp}")
            final = maskedclip.on_color(color=rgb)
        else:
            try:
                background = ImageClip(bg, duration=maskedclip.duration)
                print(f"Using {bg} as an image source to the background of {inp}")
            except Exception as e:
                print(e)
                background = VideoFileClip(bg, resize_algorithm=scaler).fx(loop, duration=maskedclip.duration)
                print(f"Using {bg} as a video source to the background of {inp}")
            final = CompositeVideoClip([background, maskedclip], size=maskedclip.size)
        audio = True
    else:
        print("No background selected, exporting the black and white mask")
        final = maskoutput
        audio = False

    outdir = config["output_dir"]
    outname = config["output_name"]
    extension = config["extension"]

    f = config["get_frame"]
    if f != 0:
        t = final.fps / f
        final.save_frame(f'{outdir}{outname}{f}.jpg', t=t)
    else:
        # https://moviepy.readthedocs.io/en/latest/_modules/moviepy/video/VideoClip.html
        filename = f'{outdir}{outname}.{extension}'
        codec = config["video_codec"]
        preset = config["compression"]
        audio_codec = config["audio_codec"]
        temp_audiofile = f'{outdir}TEMP_{outname}.mp3'
        write_logfile = config["log"]
        threads = config["threads"]
        if config["monitor"] == "gui":
            pass
        else:
            logger = config["monitor"]
        final.write_videofile(filename, codec=codec, audio=audio, preset=preset, audio_codec=audio_codec,
                              temp_audiofile=temp_audiofile, write_logfile=write_logfile, threads=threads,
                              logger=logger)


def start(config=None):
    if config is None:
        with open("config.json", "r") as read_file:
            config = json.load(read_file)
    processmovie(config)


if __name__ == '__main__':
    starttime = time.time()

    start()

    duration = time.time() - starttime
    hours = minutes = 0
    if duration > 3600:
        hours = duration / 3600
        minutes = (duration % 3600) / 60
        seconds = (duration % 3600) % 60
    elif duration > 60:
        minutes = (duration % 3600) / 60
        seconds = (duration % 3600) % 60
    else:
        seconds = duration
    print(f"Finished in {hours} hour(s), {minutes} minutes and {seconds} seconds")
