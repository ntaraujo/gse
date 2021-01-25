# Green Screen Emulator
This project is based on [Deep BGRemove](https://github.com/WhiteNoise/deep-bgremove)

![example](https://user-images.githubusercontent.com/66187211/100396165-d544a880-3022-11eb-8996-dfcf3faea716.gif)

## Current features
* Work with video/image with people
* Output a mask you can use in another editor
* Output the video/image with the background already modified by a video/image/color
* Apply a mask of your choice instead the A.I. generated one (e.g. a previous exported mask with this app)
* Use as module
* Graphical interface

## To do
* Improve time spent
* Work with more than just people images, but also objects and animals
* Complete graphical interface
* Windows executable
* Save projects to decrease processing time on multiple requests
* Improve configuration file experience and options
* Make easier the module usability

## Quickstart
Clone this repo:
```sh
git clone https://github.com/ntaraujo/gse.git
cd gse
```

If you haven't, install python in your machine. Preferably [this release](https://www.python.org/downloads/release/python-386/)

Follow [these](https://pytorch.org/get-started/locally/) instructions to install PyTorch locally (you can omit torchaudio if you wish)

E.g. the command to install the current PyTorch version for Windows and Linux with the Pip package manager and no CUDA features (not so cool):
```sh
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
Then install MoviePy and Dill
```sh
pip install moviepy dill
```

Since PyTorch cannot be installed with pip in the same way on all platforms, `requirements.txt` serves as a reference for the package versions that have been tested.

## Use with the graphical interface
Install [kivy](https://kivy.org/doc/stable/gettingstarted/installation.html) and
[kivymd](https://github.com/kivymd/KivyMD) at least 0.104.2
([why](https://stackoverflow.com/questions/61307599/filemanager-code-using-kivymd-is-not-functioning)).
E.g. on linux at the time of writing:
```sh
pip install kivy[full] https://github.com/kivymd/KivyMD/archive/master.zip
```

Run main.py:
```sh
python main.py
```

## Use with a configuration file
Rename `config.json.example` to `config.json` and edit the values to which attend your needs.
E.g. on Linux:
```sh
mv config.json.example config.json
xdg-open config.json
```

The file to run in this case is `gse.py`
```sh
python gse.py
```

## Basics of configuration file / variables
* __input__: The path to your existent video/image. The format must be supported by MoviePy. E.g. `"old_one.mp4"`

* __output_dir__: Directory for outputting videos, images, temporary files and/or logs. Need a "/" or "\\" in the end. E.g `"/home/user/Videos/"`. If `""`, defaults to current directory

* __output_name__: Part of the generated file name. This will be {output_name}.{extension} if `get_frame` is `0`. E.g. `"new_one"`

* __extension__: Video/image formats supported by MoviePy. If video, must correspond to `video_codec` and `input` must also to be a video. If image, `input` must to be an image. E.g `"mp4"` and `"jpg"`

* __background__: Path to an image or RGB color or also video if `input` is a video. If `""` the output is a black and white mask you can use in another editor. RGB colors are typed in square brackets and with commas between parameters: `[R, G, B]`. E.g. `[0, 255, 0]` (green screen) and `"path/to/my/image.jpg"`

* __relative_mask_resolution__: Depending on how big your `input` is, you may want to decrease the time spent analyzing it. This will not decrease the final output resolution but the quality and accuracy of the mask. E.g. `80` (%, percent)

* __relative_mask_fps__: For the same reasons you may want to decrease the amount of frames in your video `input` that will be computed. This will not decrease the final fps of the person in scene or even the background fps. What is affected is how fluid is the movement of the mask that accompanies the person movement. Typically, if you have a 60fps video you can use a 30fps mask without noticeable changes on the video quality. E.g. `50` (%, percent)

* __get_frame__: If you want a preview of the processing results (mainly the `relative_mask_resolution` setting) in your video `input`, set this variable to a number greater than 0. This will be the frame number exported as {output_dir}{output_name}.jpg. E.g. `535`

## Use with IPython Notebook or Python Console
You can use the features by importing the `Process` module. Can be useful if you want to save time, since when the program is run it loads a lot of stuff which will only be used with a single configuration file, a single output.
```python
from gse import Process

p = Process()  # load with a default configuration file

p = Process("config.json")  # load with a configuration file

p.load_config("config.json")  # replace the configuration file

p.relative_mask_fps = 60  # Modify a single variable

# Load the input file and let it available as p.input_clip
p.oinput()

# Change the duration to 6 seconds (for video inputs).
# See what is possible at https://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html
p.input_clip = p.input_clip.set_duration(6)

# Load the mask and let it available as p.mask_clip (p.input_clip must to exist)
p.omask()
# See type(p.mask_clip) to check the object's class

# Add a 2-second fade in effect to the mask
# See other effects in https://zulko.github.io/moviepy/ref/videofx.html
from moviepy.video.fx.fadein import fadein
p.mask_clip = p.mask_clip.fx(fadein, 2)

# Make the final clip based on the background choice and let it available as p.final_clip
# Also set p.audio which will say to the next method if the output video
# would has audio or not (p.mask_clip and p.input_clip must to exist)
p.obackground()
# See type(p.final_clip) to check the object's class

# Check the final clip duration
p.final_clip.duration

# Export to file (p.final_clip and p.audio must to exist)
p.save_file()

# Use another mask with another resolution, but with the same input
p.mask = "video_with_beautiful_shapes.mp4"
p.relative_mask_resolution = 61
p.omask()
p.obackground()
p.save_file()

# Use another background with the same mask and input
p.background = [0, 0, 255]
p.obackground()
p.save_file()

# Just export a preview of the video, with the same mask, input and background
p.get_frame = 578
p.save_file()

# In IPython Notebook you can also preview with the following, where t is measured in seconds
p.final_clip.ipython_display(t=15)

# Note that for video, the longest time is spent in p.save_file(), so there is no much to do for saving this time

# Experimental:

# Save the entire Process to use after
p.save_project("my_project.gse")

# Replace the entire current process with a saved one
p.import_project("my_project.gse")
```

## Some examples

### Masks computed with 384x216 resolution
| Example 1 | Example 2 |
| --------- | --------- |
| ![example1](https://user-images.githubusercontent.com/66187211/100396393-86e3d980-3023-11eb-90b8-06ca36d6287f.gif) | ![example2](https://user-images.githubusercontent.com/66187211/100396465-d75b3700-3023-11eb-8a34-36223b97d3ef.gif) |

### Background substitution (2160x4096)
| Original | Color | Image |
| -------- | ----- | ----- |
| ![original](https://user-images.githubusercontent.com/66187211/100396444-c4486700-3023-11eb-811a-141586f6357a.jpg) | ![color](https://user-images.githubusercontent.com/66187211/100396447-c5799400-3023-11eb-8fd7-b416821680e8.jpg) | ![image](https://user-images.githubusercontent.com/66187211/100396449-c6122a80-3023-11eb-8ad7-f1fcff976d01.jpg) |
