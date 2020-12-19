from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.filemanager import MDFileManager
import os
from kivy.properties import StringProperty, ObjectProperty, NumericProperty, BooleanProperty
from kivy.event import EventDispatcher
from kivymd.uix.picker import MDTimePicker, MDDatePicker
from datetime import datetime
from kivymd.uix.menu import MDDropdownMenu
from gse import Process
from kivy.core.window import Window
from kivymd.uix.selectioncontrol import MDCheckbox
import threading
from time import sleep
import tempfile
import shutil


class Welcome(MDScreen):
    pass


class Background(MDScreen):
    pass


class Colors(MDScreen):
    pass


class Time(MDScreen):
    pass


class Advanced(MDScreen):
    mask_menu = scaler_menu = compression_menu = video_codec_menu = audio_codec_menu = None
    video_codec_menu_items = [{"text": i} for i in ["default", "libx264 (.mp4)", "mpeg4 (.mp4)", "rawvideo (.avi)",
                                                    "png (.avi)", "libvorbis (.ogv)", "libvpx (.webm)"]]
    audio_codec_menu_items = [{"text": i} for i in ["default", "libmp3lame (.mp3)", "libvorbis (.ogg)",
                                                    "libfdk_aac (.m4a)", "pcm_s16le (.wav)", "pcm_s32le (.wav)"]]
    compression_menu_items = [{"text": i} for i in ["ultrafast", "superfast", "veryfast", "faster", "fast",
                                                    "medium", "slow", "slower", "veryslow", "placebo"]]
    scaler_menu_items = [{"text": i} for i in ["fast_bilinear", "bilinear", "bicubic", "experimental", "neighbor",
                                               "area", "bicublin", "gauss", "sinc", "lanczos", "spline"]]
    mask_menu_items = [{"text": i} for i in ["A.I.", "Video/Image"]]

    tempdir = tempfile.mkdtemp()
    frame_filename = os.path.join(tempdir, "temp_preview.jpg")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.video_codec_menu = MDDropdownMenu(
            caller=self.ids.video_codec_button,
            items=self.video_codec_menu_items,
            width_mult=4,
        )
        self.video_codec_menu.bind(on_release=self.video_codec_menu_callback)

        self.audio_codec_menu = MDDropdownMenu(
            caller=self.ids.audio_codec_button,
            items=self.audio_codec_menu_items,
            width_mult=4,
        )
        self.audio_codec_menu.bind(on_release=self.audio_codec_menu_callback)

        self.compression_menu = MDDropdownMenu(
            caller=self.ids.compression_button,
            items=self.compression_menu_items,
            width_mult=4,
        )
        self.compression_menu.bind(on_release=self.compression_menu_callback)

        self.scaler_menu = MDDropdownMenu(
            caller=self.ids.scaler_button,
            items=self.scaler_menu_items,
            width_mult=4,
        )
        self.scaler_menu.bind(on_release=self.scaler_menu_callback)

        self.mask_menu = MDDropdownMenu(
            caller=self.ids.mask_button,
            items=self.mask_menu_items,
            width_mult=4,
        )
        self.mask_menu.bind(on_release=self.mask_menu_callback)

    def video_codec_menu_callback(self, instance_menu, instance_menu_item):
        if "default" in instance_menu_item.text:
            app.ctrl.video_codec = None
        else:
            app.ctrl.video_codec = instance_menu_item.text.split()[0]
        instance_menu.caller.text = instance_menu_item.text
        instance_menu.dismiss()

    def audio_codec_menu_callback(self, instance_menu, instance_menu_item):
        if "default" in instance_menu_item.text:
            app.ctrl.audio_codec = None
        else:
            app.ctrl.audio_codec = instance_menu_item.text.split()[0]
        instance_menu.caller.text = instance_menu_item.text
        instance_menu.dismiss()

    def compression_menu_callback(self, instance_menu, instance_menu_item):
        app.ctrl.compression = instance_menu.caller.text = instance_menu_item.text
        instance_menu.dismiss()

    def scaler_menu_callback(self, instance_menu, instance_menu_item):
        app.ctrl.scaler = instance_menu.caller.text = instance_menu_item.text
        app.ctrl.do_again(1)
        instance_menu.dismiss()

    def mask_menu_callback(self, instance_menu, instance_menu_item):
        if instance_menu_item.text == "A.I.":
            app.ctrl.mask = ""
            instance_menu.caller.text = "A.I."
        else:
            app.file_manager_open()
            app.go_to = ["advanced"]
        app.ctrl.do_again(1)
        instance_menu.dismiss()

    def threads_button(self, up):
        if up:
            app.ctrl.threads += 1
        else:
            app.ctrl.threads -= 1
        self.ids.threads_label.text = f"{app.ctrl.threads} threads"


class Monitor(MDCheckbox):
    def monitor_radio(self):
        if self.active:
            app.ctrl.monitor = self.op


class Ready(MDScreen):
    pass


class Control(EventDispatcher):
    p = Process()
    conf = {'input': ["old_one.mp4", "str", False],
            'output_dir': ["", "str", False],
            'output_name': ["new_one", "str", False],
            'extension': ["mp4", "str", False],
            'video_codec': [None, "str", True],
            'audio_codec': [None, "str", True],
            'background': [[0, 255, 0], "obj", False],
            'relative_mask_resolution': [80, "num", False],
            'relative_mask_fps': [50, "num", False],
            'threads': [4, "num", False],
            'cuda': [True, "bool", False],
            'compression': ["medium", "str", False],
            'scaler': ["bicubic", "str", False],
            'monitor': ["bar", "str", True],
            'log': [False, "bool", False],
            'get_frame': [0, "num", False],
            'fake_get_frame': [0, "num", False],
            'mask': ["", "str", False]}

    for key in conf:
        if conf[key][1] == "str":
            func = StringProperty
        elif conf[key][1] == "num":
            func = NumericProperty
        elif conf[key][1] == "bool":
            func = BooleanProperty
        else:
            func = ObjectProperty
        exec(f"""
{key} = func(conf[key][0], allownone=conf[key][2])
def on_{key}(self, instance, value):
    self.p.{key} = value
    c = f'"{{value}}"' if type(value) == str else str(value)
    print(f'{key} changed to {{c}}')""")

    ps = [p.oinput, p.omask, p.obackground, p.save_file]
    doing = [False for _ in ps]
    done = [False for _ in ps]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cs = [self.call_input, self.call_mask, self.call_background, self.call_save]

    def do_again(self, n):
        max = len(self.ps)
        print(f"Scheduling processes from {n} until {max}")
        for x in range(n, max):
            self.done[x] = self.doing[x] = False

    def call(self, n):
        threading.Thread(target=self.cs[n], daemon=True).start()

    def base_call(self, n):
        if not self.done[n]:
            self.doing[n] = True
            print(f"Process {n} started")
            self.ps[n]()
            self.done[n] = True
            print(f"Process {n} finished")

    def base_check(self, needed, next):
        if not self.doing[needed]:
            self.call(needed)
        while not self.done[needed]:
            sleep(1)
        self.base_call(next)

    def call_input(self):
        self.base_call(0)
        self.do_again(1)

    def call_mask(self):
        self.base_check(0, 1)
        self.do_again(2)

    def call_background(self):
        self.base_check(1, 2)
        self.do_again(3)

    def call_save(self):
        self.base_check(2, 3)


class GSE(MDApp):
    sm = None
    go_to = ["welcome"]
    ctrl = Control()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
        )

    def ready_output(self, inst):
        strs = inst.text.split(".")
        if len(strs) == 2:
            inst.hint_text = inst.text
            self.ctrl.output_name = strs[0]
            self.ctrl.extension = strs[1]
        else:
            pass  # error

    def file_manager_open(self):
        # parent = os.path.dirname(os.path.abspath(os.getcwd()))
        home = os.path.expanduser("~")
        self.file_manager.show(home)
        self.manager_open = True

    def select_path(self, path):
        self.exit_manager()
        if self.sm.current == "welcome":
            self.ctrl.input = path
            self.ctrl.do_again(0)
        elif self.sm.current == "background":
            self.ctrl.background = path
            self.ctrl.do_again(2)
        elif self.sm.current == "ready":
            self.ctrl.output_dir = os.path.join(path, "")
        elif self.sm.current == "advanced":
            self.ctrl.mask = path

        self.change()

    def exit_manager(self, *args):
        self.manager_open = False
        self.file_manager.close()

    def show_date_picker(self):
        date_dialog = MDDatePicker(callback=self.get_date)
        date_dialog.open()

    def get_date(self, date):
        self.show_time_picker()

    def show_time_picker(self):
        time_dialog = MDTimePicker()
        time_dialog.bind(time=self.get_time)
        previous_time = datetime.now()
        time_dialog.set_time(previous_time)
        time_dialog.open()

    def get_time(self, instance, time):
        self.change()
        return time

    def change(self, to=None):
        if to:
            print(f"Changing to {to} screen")
            self.sm.current = to
        elif len(self.go_to) > 0:
            print(f"Changing to {self.go_to[-1]}")
            self.sm.current = self.go_to[-1]
            self.go_to.pop()

    def build(self):
        self.theme_cls.primary_palette = "LightGreen"
        # self.theme_cls.theme_style = "Dark"
        self.sm = ScreenManager()
        self.sm.add_widget(Welcome())
        self.sm.add_widget(Background())
        self.sm.add_widget(Colors())
        self.sm.add_widget(Time())
        self.sm.add_widget(Ready())
        self.advanced = Advanced()
        self.sm.add_widget(self.advanced)

        self.change()

        return self.sm


if __name__ == "__main__":
    Window.maximize()
    app = GSE()
    app.run()
    shutil.rmtree(app.advanced.tempdir)
