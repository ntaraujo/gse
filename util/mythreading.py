from threading import Thread
from traceback import format_exc


class MyThread(Thread):

    def __init__(self, target, args=None, kwargs=None, exc_callback=None, daemon=True, *t_args, **t_kwargs):
        super().__init__(daemon=daemon, *t_args, **t_kwargs)
        self.target, self.args, self.kwargs = target, list(), dict()
        if exc_callback:
            self.exc_callback = exc_callback
        if args:
            self.args = args
        if kwargs:
            self.kwargs = kwargs

    @staticmethod
    def exc_callback(target, args, kwargs, traceback):
        arg_txt = f"{args} as arguments" if args else "no arguments"
        kwarg_txt = f"{kwargs} as key word arguments" if kwargs else "no key word arguments"
        print(f"Exception in {target} with {arg_txt} and {kwarg_txt}:\n{traceback}")

    def run(self):
        try:
            self.target(*self.args, **self.kwargs)
        except Exception:
            traceback = format_exc()
            self.exc_callback(self.target, self.args, self.kwargs, traceback)
