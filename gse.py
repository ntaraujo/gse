from torch.hub import load as hub_load
from torch import FloatTensor, no_grad, tanh, ones_like
from torch.cuda import is_available as cuda_available
from torch.nn.functional import relu, conv2d, hardtanh
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from torchvision.transforms import Compose, Normalize
from json import load as jload
from json import dump as jdump
from moviepy.video.fx.loop import loop
from moviepy.video.fx.resize import resize
from time import perf_counter
from imghdr import what as is_image
from dill import dump as ddump
from dill import load as dload
from os.path import dirname, basename, splitext, abspath
from os.path import join as join_path
from ast import literal_eval
from typing import Union, Optional, Callable, Any, IO, Iterable, NewType, List
from os import PathLike
from numpy import ndarray


class MakeMask:
    def __init__(self, cuda: bool):
        """
        Loads the needed to run once for transforming frames with __call__ \n
        E.g.
            mm = MakeMask(True) \n
            new_image = mm(old_image)

        :param cuda: should the process occur on Nvidia GPU?
        """

        self.cuda = cuda
        self.model = hub_load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
        self.people_class = 15

        self.model.eval()
        print("Model Loaded")

        self.blur = FloatTensor([[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]]) / 16.0

        # move the input and model to GPU for speed if available ?
        if self.cuda and cuda_available():
            print("Using GPU (CUDA) to process the images")
            self.model.to('cuda')
            self.blur = self.blur.to('cuda')

        self.preprocess = Compose(
            [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    def __call__(self, img: ndarray) -> ndarray:
        """
        Transform a given frame to a black and white one, representing a mask for editors \n
        E.g.
            mm = MakeMask(True) \n
            new_image = mm(old_image)
        """
        frame_data = FloatTensor(img) / 255.0

        input_tensor = self.preprocess(frame_data.permute(2, 0, 1))
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available ?
        if self.cuda and cuda_available():
            input_batch = input_batch.to('cuda')

        with no_grad():
            output = self.model(input_batch)['out'][0]

        segmentation = output.argmax(0)

        bgout = output[0:1][:][:]
        a = (1.0 - relu(tanh(bgout * 0.30 - 1.0))).pow(0.5) * 2.0

        people = segmentation.eq(ones_like(segmentation).long().fill_(self.people_class)).float()

        people.unsqueeze_(0).unsqueeze_(0)

        for i in range(3):
            people = conv2d(people, self.blur, stride=1, padding=1)

        # combined_mask = tnf.hardtanh(a * b)
        combined_mask = relu(hardtanh(a * (people.squeeze().pow(1.5))))
        combined_mask = combined_mask.expand(1, 3, -1, -1)

        newimg = (combined_mask * 255.0).cpu().squeeze().byte().permute(1, 2, 0).numpy()

        return newimg


ClipType = Union[VideoFileClip, ImageClip]
FinalClipType = NewType('FinalClipType', Union[ClipType, CompositeVideoClip])
PathType = Union[str, bytes, PathLike]


def get_input_clip(input: PathType, **videofileclip_args) -> ClipType:
    """
    Returns a moviepy clip for using with gse \n
    E.g.
        input_clip = get_input_clip("video.mp4") \n
        mask_clip = get_mask_clip(input_clip)

    :param input: video/image path
    :param videofileclip_args: additional arguments for moviepy.video.io.VideoFileClip.__init__
    """
    if is_image(input):
        print(f"Loading {input} as the main image source")
        return ImageClip(input, duration=1).set_fps(1)
    else:
        print(f"Loading {input} as the main video source")
        return VideoFileClip(input, **videofileclip_args)


def get_mask_clip(input_clip: ClipType, relative_mask_fps: int = 100, relative_mask_resolution: int = 100,
                  mask: PathType = "", cuda: bool = True, **videofileclip_args) -> ClipType:
    """
    Returns a moviepy clip with the right attributes to be used as mask for the input_clip \n
    E.g.
        mask_clip = get_mask_clip(input_clip) \n
        final_clip = get_final_clip(mask_clip, input_clip, [0, 255, 0])

    :param input_clip: got with gse.get_input_clip
    :param relative_mask_fps: percentage. How fluid is the movement of the mask that accompanies the person movement?
    :param relative_mask_resolution: percentage. The quality and accuracy of the mask
    :param mask: if you want to use a saved mask video instead the A.I. generated one
    :param cuda: should part of the process occur on Nvidia GPU?
    :param videofileclip_args: additional arguments for moviepy.video.io.VideoFileClip.__init__
    """
    if mask != "":  # if given
        if is_image(mask):
            print(f"Loading the image {mask} as the mask for {input_clip.filename}")
            return ImageClip(mask, duration=input_clip.duration)
        else:
            print(f"Loading the video {mask} as the mask for {input_clip.filename}")
            return VideoFileClip(mask, **videofileclip_args) \
                .fx(loop, duration=input_clip.duration).set_duration(input_clip.duration)
    else:  # if should be result of A.I.
        process_clip = input_clip.copy()
        fps = relative_mask_fps * 0.01
        if fps != 1:  # if asked to change fps
            newfps = input_clip.fps * fps
            process_clip = process_clip.set_fps(newfps)
            print(f"Mask fps decreased in {(1 - fps) * 100}%. {process_clip.fps}fps now")
        res = relative_mask_resolution * 0.01
        if res != 1:  # if asked to resize
            process_clip = process_clip.fx(resize, res)
            w, h = process_clip.size
            print(f"Mask resolution decreased in {(1 - res) * 100}%, {w}x{h} now")
        return process_clip.fl_image(MakeMask(cuda))


def get_final_clip(mask_clip: ClipType, input_clip: ClipType, background: Union[List[float], PathType],
                   **videofileclip_args) -> FinalClipType:
    """
    Apply the mask_clip to the input_clip and use the background, in a way to be used with gse.save_to_file
    or simply returns the mask_clip \n
    E.g.
        final_clip = get_final_clip(mask_clip, input_clip, [0, 255, 0]) \n
        save_to_file(final_clip, "path/video.mp4")

    :param mask_clip: got with gse.get_mask_clip
    :param input_clip: got with gse.get_input_clip
    :param background: color [R, G, B] or path to video/image or empty string, so the mask_clip is directly returned
    :param videofileclip_args: additional arguments for moviepy.video.io.VideoFileClip.__init__
    """
    if background != "":
        usable_mask = mask_clip.fx(resize, input_clip.size).to_mask()
        masked_clip = input_clip.set_mask(usable_mask)
        if type(background) == list:  # if color
            rgb = (background[0], background[1], background[2])
            print(f"Using the RGB color {rgb} as the background of {input_clip.filename}")
            to_return = masked_clip.on_color(color=rgb)
        elif is_image(background):
            print(f"Using {background} as image source to the background of {input_clip.filename}")
            background_clip = ImageClip(background, duration=masked_clip.duration)
            to_return = smooth_composite(background_clip, masked_clip)
        else:
            print(f"Using {background} as video source to the background of {input_clip.filename}")
            background_clip = VideoFileClip(background, **videofileclip_args) \
                .fx(loop, duration=masked_clip.duration).set_duration(input_clip.duration)
            to_return = smooth_composite(background_clip, masked_clip)
        to_return.filename = input_clip.filename
        return to_return
    else:
        print("No background selected, skipping compositing")
        return mask_clip


def smooth_composite(back: ClipType, front: ClipType):
    """
    Composite two clips, one in back and other in front, resizing the back as needed \n
    E.g.
        new_video = smooth_composite(clip_with_big_size, clip_with_right_size) \n
        print(new_video.size == clip_with_right_size.size) \n
        # True

    :param back: a moviepy clip
    :param front: a moviepy clip
    """
    wf, hf = front.size
    wb, hb = back.size
    rf = wf / hf
    rb = wb / hb
    if rf > rb:
        back = back.fx(resize, width=wf)
    else:
        back = back.fx(resize, height=hf)
    return CompositeVideoClip([back, front.set_position("center")], size=front.size)


def save_to_file(final_clip: FinalClipType, output: Optional[PathType], get_frame_from_time: int = 0,
                 get_frame: int = 0, alpha: bool = False, output_dir: Optional[PathType] = None,
                 output_name: Optional[str] = None, extension: Optional[str] = None, **write_videofile_args):
    """
    Write a moviepy clip with the attribute clip.filename as video or image if the filename refers to an image or if
    asked explicitly \n
    E.g.
        save_to_file(final_clip, "path/video.mp4") \n
        from IPython.display import Video \n
        Video("path/video.mp4")

    :param final_clip: got with gse.get_final_clip
    :param output: where and with what name and extension the clip should be saved
    :param get_frame_from_time: if you want to extract the frame at X seconds
    :param get_frame: if you want to extract the X° frame
    :param alpha: if image, should keep the alpha channel (transparency, .png)?
    :param write_videofile_args: additional arguments for moviepy.video.VideoClip.VideoClip.write_videofile
    :param output_dir: for compatibility
    :param output_name: for compatibility
    :param extension: for compatibility
    """
    if output_dir:
        output = abspath(join_path(output_dir, f'{output_name}.{extension}'))

    if is_image(final_clip.filename) or get_frame or get_frame_from_time:
        if get_frame:
            get_frame_from_time = final_clip.fps / get_frame
        elif not get_frame_from_time:
            get_frame_from_time = final_clip.duration / 2
        print(f'Saving as image to {output}')
        final_clip.save_frame(output, t=get_frame_from_time, withmask=alpha)
    else:
        temp_audiofile = abspath(join_path(dirname(output), splitext(basename(output))[0] + '.mp3'))
        final_clip.write_videofile(output, temp_audiofile=temp_audiofile, **write_videofile_args)


class Project:
    def __init__(self, config: Optional[PathType] = None):
        """
        Define variables and optionally loads a project file \n
        E.g.
            p = Project("config.json")

        :param config: path to a .json or .gse file
        """
        self.input_clip = self.mask_clip = self.final_clip = None
        self.audio = True
        if config:
            self.load(config)

    def var(self, var: str, converter: Union[type, str, None] = None, asker: Callable[[str], Any] = input) -> Any:
        """
        Verify if a gse.Project variable exists before calling it. If doesn't, try to obtain it and optionally convert
        to a specific type. The function does not create the variable. \n
        E.g.
            another_var = p.var("some_var", str, lambda var_name: var_name + ' not found') \n
            another_var = p.var("some_var", "auto") \n
            another_var = p.var("some_var", asker=lambda _: None)

        :param var: The variable name as string
        :param converter: if None, no converting, if "auto" find the probably right type, if a type e.g. bool, use it
        :param asker: function which returns a value for the given variable name, if the variable doesn't exist
        """
        if var in self.__dict__.keys():
            return self.__dict__[var]
        if asker == input:
            to_return = input(f'Variable {var}: ')
        else:
            to_return = asker(var)
        if not converter:
            return to_return
        elif converter == "auto":
            try:
                return literal_eval(to_return)
            except (ValueError, SyntaxError):
                return to_return
        else:
            try:
                return converter(to_return)
            except ValueError:
                return to_return

    @staticmethod
    def serialize(obj):
        """
        Return a default value for a non-serializable object \n
        E.g.
            print(p.serialize(lambda: None)) \n
            # <<non-serializable function>>
        """
        return f'<<non-serializable {type(obj).__qualname__}>>'

    def save(self, path: Union[IO[str], PathType]) -> None:
        """
        Save the gse.Project objects to a file, completely as a ".gse" or partially as a ".json" \n
        E.g.
            p.save("config.json") \n
            p.load("config.json")

        :param path: directory, base name and extension of file to save
        """
        file_type = splitext(path)[1]
        with open(f'{path}', "wb") as project_file:
            if file_type == ".gse":
                ddump(self, project_file)
            elif file_type == ".json":
                jdump(self.__dict__, path, default=self.serialize)
                print(f'Attention: .json projects do not keep non-serializable variables.')
            else:
                raise Exception(f'Impossible to load file with extension "{file_type}". Accepted: ".gse" and ".json"')
        print(f'Saved to {path}\n{self.__dict__}')

    def load(self, path: PathType) -> None:
        """
        Load objects to gse.Project class from a file, completely with a ".gse" or partially with a ".json" \n
        E.g.
            type(p.input_clip).__qualname__ \n
            # NoneType \n
            p.load("project.gse") \n
            type(p.input_clip).__qualname__ \n
            # VideoFileClip

        :param path: path to the file to load from
        """
        file_type = splitext(path)[1]
        with open(path, "rb") as project_file:
            if file_type == ".gse":
                self.__dict__.update(dload(project_file).__dict__)
            elif file_type == ".json":
                for var_name, value in jload(project_file).items():
                    if var_name[0] == '_' or (type(value) == str and value[:18] == '<<non-serializable'):
                        pass
                    else:
                        self.__dict__[var_name] = value
            else:
                raise Exception(f'Impossible to load file with extension "{file_type}". Accepted: ".gse" and ".json"')

    def processes(self, processes: Iterable[int] = range(4), asker: Callable[[Any], Any] = input, **update_args):
        """
        Run gse functions in a default way, according to gse.Project
        configuration variables, but allowing to modify pieces of the process.

        function 0: gse.get_input_clip \n
        function 1: gse.get_mask_clip \n
        function 2: gse.get_final_clip \n
        function 3: gse.save_to_file

        E.g.
            p.processes() \n
            p.processes([1], write_logfile=True) \n
            p.processes(range(3), lambda _: None) \n

        :param processes: list, iterable with the number(s) of desired function(s)
        :param asker: parameter of gse.Project.var
        :param update_args: optional arguments to overwrite default ones in each function
        """
        def var(name: str, converter: Union[type, str, None]):
            return self.var(name, converter, asker)

        def compatibility(wrong: str, right: str):
            if wrong in update_args:
                update_args[right] = update_args[wrong]
                del update_args[wrong]

        compatibility("scaler", "resize_algorithm")
        compatibility("compression", "preset")
        compatibility("log", "write_logfile")

        if 0 in processes:
            args = {"input": var("input", str),
                    "resize_algorithm": var("scaler", str)}

            self.input_clip = get_input_clip(**args)
        if 1 in processes:
            args = {"input_clip": self.input_clip,
                    "relative_mask_fps": var("relative_mask_fps", int),
                    "relative_mask_resolution": var("relative_mask_resolution", int),
                    "mask": var("mask", str),
                    "cuda": var("cuda", bool),
                    "resize_algorithm": var("scaler", str)}

            args.update(update_args)

            self.mask_clip = get_mask_clip(**args)
        if 2 in processes:
            args = {"mask_clip": self.mask_clip,
                    "input_clip": self.input_clip,
                    "background": var("background", "auto"),
                    "resize_algorithm": var("scaler", str)}

            args.update(update_args)

            self.final_clip = get_final_clip(**args)
        if 3 in processes:
            file = '.'.join([var("output_name", str), var("extension", str)])
            path = abspath(join_path(var("output_dir", str), file))
            if var("background", "auto") == "":
                self.audio = False

            args = {"clip": self.final_clip,
                    "output": path,
                    "get_frame": var("get_frame", int),
                    "preset": var("compression", str),
                    "audio": self.audio,
                    "write_logfile": var("log", bool),
                    "threads": var("threads", int)}
            if var("video_codec", "auto"):
                args["codec"] = var("video_codec", str)
            if var("audio_codec", "auto"):
                args["audio_codec"] = var("audio_codec", str)

            args.update(update_args)

            save_to_file(**args)


class Timer:
    def __init__(self):
        self.hours = self.minutes = self.seconds = self.starttime = self.stoptime = 0
        self.start()

    def start(self):
        self.starttime = self.stoptime = perf_counter()

    def stop(self):
        self.stoptime = perf_counter()

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

    p = Project("config.json")
    p.processes()

    t.finish()
