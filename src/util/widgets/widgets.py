from kivymd.uix.list import OneLineIconListItem
from kivy.properties import StringProperty, NumericProperty
from kivymd.uix.navigationdrawer import MDNavigationDrawer
from kivymd.uix.stacklayout import MDStackLayout
from kivy.lang.builder import Builder
from os.path import dirname, abspath, join
from glob import glob


class ItemDrawer(OneLineIconListItem):
    icon = StringProperty()
    to_screen = StringProperty()


class LeftMenu(MDNavigationDrawer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        icons_item = {
            "home": ["Home page", "welcome"],
            "folder-multiple-image": ["Choose the background", "background"],
            "clock-check-outline": ["Processing time", "time"],
            "export": ["Export", "ready"],
            "cog-outline": ["Advanced options", "advanced"],
        }
        for icon_name in icons_item.keys():
            self.ids.drawer_list.add_widget(
                ItemDrawer(icon=icon_name, text=icons_item[icon_name][0], to_screen=icons_item[icon_name][1])
            )


class CenteredStackLayout(MDStackLayout):
    bigchild = NumericProperty()

    def do_layout(self, *args):
        super().do_layout(*args)
        self.set_bigchild()

    def set_bigchild(self):
        bigchild = 0
        for child in self.children:
            bigchild += child.width
        self.bigchild = bigchild + self.spacing[1] * 9


this_dir = dirname(abspath(__file__))
kvs = glob(join(this_dir, '*.kv'))

for kv in kvs:
    Builder.load_file(kv)
