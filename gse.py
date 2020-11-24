import torch
import torch.nn.functional as tnf
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from torchvision import transforms
import json
from moviepy.video.fx.loop import loop
from moviepy.video.fx.resize import resize
import time
import imghdr


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


class Process:
    def __init__(self, config):  # set variables
        self.inp = config["input"]
        self.scaler = config["scaler"]
        self.premask = config["mask"]
        self.res = config["relative_mask_resolution"] * 0.01
        self.fps = config["relative_mask_fps"] * 0.01
        self.cuda = config["cuda"]
        self.bg = config["background"]
        self.dir = config["output_dir"]
        self.name = config["output_name"]
        self.ext = config["extension"]
        self.f = config["get_frame"]
        self.logger = config["monitor"]
        self.audio_codec = config["audio_codec"]
        self.write_logfile = config["log"]
        self.threads = config["threads"]
        self.filename = f'{self.dir}{self.name}.{self.ext}'
        self.temp_audiofile = f'{self.dir}TEMP_{self.name}.mp3'
        self.codec = config["video_codec"]
        self.preset = config["compression"]
        self.imgext = "jpg"

    def get_input(self):
        if not imghdr.what(self.inp):  # if video
            print(f"Loading {self.inp} as the main video source")
            clip = VideoFileClip(self.inp, resize_algorithm=self.scaler)
        else:  # if image
            print(f"Loading {self.inp} as the main image source")
            clip = ImageClip(self.inp, duration=0.1)
            self.f = 1
            self.imgext = self.ext
        return clip

    def get_mask(self, clip):
        if self.premask != "":  # if given
            if imghdr.what(self.premask):  # image
                print(f"Loading the image {self.premask} as the mask for {self.inp}")
                mask = ImageClip(self.premask, duration=clip.duration)
            else:  # video
                print(f"Loading the video {self.premask} as the mask for {self.inp}")
                mask = VideoFileClip(self.premask, resize_algorithm=self.scaler)
        else:  # if result of A.I.
            processclip = clip.copy()
            if self.res != 1:  # resize if asked
                processclip = self.resize(processclip, self.res, self.scaler)
            if self.fps != 1:  # change fps if asked
                processclip = self.refps(processclip, self.fps)
            mask = processclip.fl_image(MakeMask(self.cuda))
        return mask

    def apply(self, clip, mask):  # if background
        if self.bg == "":  # no exists, skip compositing
            print("No background selected, exporting the black and white mask")
            final = mask
            audio = False
        else:  # exists, use
            maskedclip = clip.set_mask(mask.to_mask())
            if type(self.bg) == list:  # if color
                rgb = (self.bg[0], self.bg[1], self.bg[2])
                print(f"Using the RGB color {rgb} as the background of {self.inp}")
                final = maskedclip.on_color(color=rgb)
            elif imghdr.what(self.bg):  # if image
                print(f"Using {self.bg} as image source to the background of {self.inp}")
                bg = ImageClip(self.bg, duration=maskedclip.duration)
                final = self.composite(bg, maskedclip)
            else:  # if video
                print(f"Using {self.bg} as video source to the background of {self.inp}")
                bg = VideoFileClip(self.bg, resize_algorithm=self.scaler).fx(loop, duration=maskedclip.duration)
                final = self.composite(bg, maskedclip)
            audio = True
        return final, audio

    def save(self, clip, audio):  # save
        if self.f != 0:  # an image
            flname = f'{self.dir}{self.name}.{self.imgext}'
            t = self.f / clip.fps
            clip.save_frame(flname, t=t, withmask=False)
        else:  # the video
            if self.logger == "gui":
                self.logger = "proglog here"
            clip.write_videofile(self.filename, codec=self.codec, audio=audio, preset=self.preset,
                                 audio_codec=self.audio_codec, temp_audiofile=self.temp_audiofile,
                                 write_logfile=self.write_logfile, threads=self.threads, logger=self.logger)

    def all(self):
        i = self.get_input()
        m = self.get_mask(i)
        c, a = self.apply(i, m)
        self.save(c, a)

    def resize(self, clip, decimal, scaler):
        oldx, oldy = clip.size
        newx = int(oldx * decimal)
        newy = int(oldy * decimal)
        print(f"Decreasing mask resolution in {(1 - decimal) * 100}%. {newx}x{newy} now")
        return clip.fx(resize, decimal, method=scaler)

    def refps(self, clip, decimal):
        newfps = clip.fps * decimal
        print(f"Decreasing mask fps in {(1 - decimal) * 100}%. {newfps}fps now")
        return clip.set_fps(newfps)

    def composite(self, back, front):
        return CompositeVideoClip([back, front], size=front.size)


class Timer:
    def __init__(self):
        self.hours = self.minutes = self.seconds = self.starttime = self.stoptime = 0
        self.start()

    def start(self):
        self.starttime = self.stoptime = time.time()

    def stop(self):
        self.stoptime = time.time()

    def sec_duration(self):
        return self.stoptime - self.starttime

    def set_hours(self):
        duration = self.sec_duration()
        if duration > 3600:
            self.hours = duration / 3600
            self.minutes = (duration % 3600) / 60
            self.seconds = (duration % 3600) % 60
        elif duration > 60:
            self.minutes = (duration % 3600) / 60
            self.seconds = (duration % 3600) % 60
        else:
            self.seconds = duration

    def print_time(self):
        print(f"Finished in {int(self.hours)} hour(s), {int(self.minutes)} minute(s) and {int(self.seconds)} second(s)")

    def finish(self):
        self.stop()
        self.set_hours()
        self.print_time()


if __name__ == '__main__':
    t = Timer()

    with open("config.json", "r") as read_file:
        config = json.load(read_file)
    p = Process(config)
    p.all()

    t.finish()
