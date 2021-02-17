from kivymd.uix.screen import MDScreen
from kivy.lang.builder import Builder
from os.path import dirname, abspath, join
from glob import glob


class Welcome(MDScreen):
    pass


class Background(MDScreen):
    pass


class Colors(MDScreen):
    pass


class Time(MDScreen):
    pass


class Ready(MDScreen):
    pass


this_dir = dirname(abspath(__file__))
kvs = glob(join(this_dir, '*.kv'))

for kv in kvs:
    Builder.load_file(kv)
