from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.filemanager import MDFileManager
from kivymd.toast import toast
from kivy.core.window import Window
import os
from gse import Process


class Manager(ScreenManager):
    pass


class Welcome(MDScreen):
    pass


class Background(MDScreen):
    pass


class Time(MDScreen):
    pass


class Advanced(MDScreen):
    pass


class GSE(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.manager_open = False
        Window.bind(on_keyboard=self.events)
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
        )

    def file_manager_open(self):
        self.file_manager.show(os.path.abspath(os.getcwd()))  # output manager to the screen
        self.manager_open = True

    def select_path(self, path):
        self.exit_manager()
        toast(path)

    def exit_manager(self, *args):
        self.manager_open = False
        self.file_manager.close()

    def events(self, instance, keyboard, keycode, text, modifiers):
        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True

    def build(self):
        self.theme_cls.primary_palette = "LightGreen"
        # self.theme_cls.theme_style = "Dark"
        return Manager()


if __name__ == '__main__':
    GSE().run()
