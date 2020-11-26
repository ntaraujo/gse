import torch
import torch.nn.functional as tnf
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from torchvision import transforms
import json
from moviepy.video.fx.loop import loop
import time
import imghdr
import dill


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
    def __init__(self, conf=None):  # set variables
        if not conf:
            config = {'input': 'old_one.mp4',
                      'output_dir': '',
                      'output_name': 'new_one',
                      'extension': 'mp4',
                      'video_codec': None,
                      'audio_codec': None,
                      'background': [0, 255, 0],
                      'relative_mask_resolution': 80,
                      'relative_mask_fps': 50,
                      'threads': 4,
                      'cuda': True,
                      'compression': 'medium',
                      'scaler': 'bicubic',
                      'monitor': 'bar',
                      'log': False,
                      'get_frame': 0,
                      'mask': ''}
        else:
            config = conf

        self.input = config["input"]
        self.scaler = config["scaler"]
        self.mask = config["mask"]
        self.relative_mask_resolution = config["relative_mask_resolution"]
        self.relative_mask_fps = config["relative_mask_fps"]
        self.cuda = config["cuda"]
        self.background = config["background"]
        self.output_dir = config["output_dir"]
        self.output_name = config["output_name"]
        self.extension = config["extension"]
        self.monitor = config["monitor"]
        self.audio_codec = config["audio_codec"]
        self.log = config["log"]
        self.threads = config["threads"]
        self.video_codec = config["video_codec"]
        self.compression = config["compression"]
        self.get_frame = config["get_frame"]

        self.res = self.relative_mask_resolution * 0.01
        self.fps = self.relative_mask_fps * 0.01
        self.filename = f'{self.output_dir}{self.output_name}.{self.extension}'
        self.temp_audiofile = f'{self.output_dir}TEMP_{self.output_name}.mp3'
        if self.monitor == "gui":
            self.logger = "bar"  # yet
        else:
            self.logger = self.monitor
        if self.input != "old_one.mp4" and not imghdr.what(self.input):
            self.imgext = "jpg"
            self.f = self.get_frame
        else:
            self.imgext = self.extension
            self.f = 1

    def oinput(self):
        if not imghdr.what(self.input):  # if video
            print(f"Loading {self.input} as the main video source")
            self.input_clip = VideoFileClip(self.input, resize_algorithm=self.scaler)
        else:  # if image
            print(f"Loading {self.input} as the main image source")
            self.input_clip = ImageClip(self.input, duration=1).set_fps(1)

    def omask(self):
        if self.mask != "":  # if given
            if imghdr.what(self.mask):  # image
                print(f"Loading the image {self.mask} as the mask for {self.input}")
                self.mask_clip = ImageClip(self.mask, duration=self.input_clip.duration)
            else:  # video
                print(f"Loading the video {self.mask} as the mask for {self.input}")
                self.mask_clip = VideoFileClip(self.mask, resize_algorithm=self.scaler).fx(loop, duration=self.input_clip.duration)
        else:  # if result of A.I.
            processclip = self.input_clip.copy()
            if self.fps != 1:  # if asked to change fps
                newfps = self.input_clip.fps * self.fps
                processclip = processclip.set_fps(newfps)
                print(f"Mask fps decreased in {(1 - self.fps) * 100}%. {processclip.fps}fps now")
            if self.res != 1:  # if asked to resize
                processclip = processclip.resize(self.res)
                w = processclip.size[0]
                h = processclip.size[1]
                print(f"Mask resolution decreased in {(1 - self.res) * 100}%, {w}x{h} now")
            self.mask_clip = processclip.fl_image(MakeMask(self.cuda))

    def obackground(self):  # if background
        if self.background == "":  # no exists
            print("No background selected, skipping compositing")
            self.final_clip = self.mask_clip
            self.audio = False
        else:  # exists
            usable_mask = self.mask_clip.resize(newsize=self.input_clip.size).to_mask()
            maskedclip = self.input_clip.set_mask(usable_mask)
            if type(self.background) == list:  # if color
                rgb = (self.background[0], self.background[1], self.background[2])
                print(f"Using the RGB color {rgb} as the background of {self.input}")
                self.final_clip = maskedclip.on_color(color=rgb)
            elif imghdr.what(self.background):  # if image
                print(f"Using {self.background} as image source to the background of {self.input}")
                bg = ImageClip(self.background, duration=maskedclip.duration)
                self.final_clip = self.composite(bg, maskedclip)
            else:  # if video
                print(f"Using {self.background} as video source to the background of {self.input}")
                bg = VideoFileClip(self.background, resize_algorithm=self.scaler).fx(loop, duration=maskedclip.duration)
                self.final_clip = self.composite(bg, maskedclip)
            self.audio = True

    def save_file(self):  # save
        if self.f != 0:  # an image
            flname = f'{self.output_dir}{self.output_name}.{self.imgext}'
            tim = self.f / self.final_clip.fps
            print("Saving as image")
            self.final_clip.save_frame(flname, t=tim, withmask=False)
        else:  # the video
            print("Saving as video")
            self.final_clip.write_videofile(self.filename, codec=self.video_codec, audio=self.audio,
                                            preset=self.compression, audio_codec=self.audio_codec,
                                            temp_audiofile=self.temp_audiofile, write_logfile=self.log,
                                            threads=self.threads, logger=self.logger)

    def save_project(self, project_file):
        with open(f'{project_file}', "wb") as project:
            dill.dump(self, project)

    def import_project(self, project_file):
        with open(project_file, "rb") as project_file:
            self.__dict__.update(dill.load(project_file).__dict__)

    def load_config(self, config):
        if type(config) == dict:
            conf = config
            self.__init__(config)
        else:
            with open(config, "r") as read_file:
                conf = json.load(read_file)
        self.__init__(conf)
        return conf

    def all(self):
        self.oinput()
        self.omask()
        self.obackground()
        self.save_file()

    def composite(self, back, front):
        wf, hf = front.size
        wb, hb = back.size
        rf = wf / hf
        rb = wb / hb
        if rf > rb:
            back = back.resize(width=wf)
        else:
            back = back.resize(height=hf)
        return CompositeVideoClip([back, front.set_position("center")], size=front.size)


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

    p = Process()
    p.load_config("config.json")
    p.all()

    t.finish()
