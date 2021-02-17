from threading import Thread
from traceback import format_exc


class MyThread(Thread):

    def __init__(self, target, args=None, exc_callback=None, daemon=True, *idont, **know):
        super().__init__(daemon=daemon, *idont, **know)
        self.target, self.args = target, args
        if exc_callback:
            self.exc_callback = exc_callback

    @staticmethod
    def exc_callback(target, args, traceback):
        arg_txt = f"{args} as arguments" if args else "no arguments"
        print(f"Exception in {target} with {arg_txt}:\n{traceback}")

    def run(self):
        try:
            if self.args:
                self.target(*self.args)
            else:
                self.target()
        except Exception:
            traceback = format_exc()
            self.exc_callback(self.target, self.args, traceback)
