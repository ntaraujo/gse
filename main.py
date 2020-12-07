from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.app import App
from gse import Process
from kivy.uix.label import Label

class Manager(ScreenManager):
    pass

class Welcome(Screen):
    pass

class Background(Screen):
    pass

class Time(Screen):
    pass

class Advanced(Screen):
    pass

class GSE(App):
    def build(self):
        return Label(text="roi")

if __name__ == '__main__':
    GSE().run()