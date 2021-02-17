from threading import Thread
from traceback import format_exc
from typing import Callable, Optional, Any, Iterable, Mapping


target_type = Callable[[Any], None]
kwargs_type = Mapping[str, Any]


class MyThread(Thread):

    def __init__(self, target: target_type, args: Optional[Iterable] = None, kwargs: Optional[kwargs_type] = None,
                 exc_callback: Callable[[target_type, Iterable, kwargs_type, str], None] = None,
                 daemon: bool = True, *t_args, **t_kwargs):
        """
        Thread but handle exceptions with exc_callback

        :param target: function will run with MyThread.start
        :param args: arguments to target
        :param kwargs: key word arguments to target
        :param exc_callback: function receiving target, args, kwargs and the traceback of any error
        :param daemon: stop thread when code finishes?
        :param t_args: arguments for Thread
        :param t_kwargs: key word arguments for Thread
        """
        super().__init__(daemon=daemon, *t_args, **t_kwargs)
        self.target, self.args, self.kwargs = target, list(), dict()
        self.exc_callback = exc_callback if exc_callback else default_exc_callback
        if args:
            self.args = args
        if kwargs:
            self.kwargs = kwargs

    def run(self):
        try:
            self.target(*self.args, **self.kwargs)
        except Exception:
            traceback = format_exc()
            self.exc_callback(self.target, self.args, self.kwargs, traceback)


def default_exc_callback(target: target_type, args: Iterable, kwargs: kwargs_type, traceback: str) -> None:
    func = f'{target.__name__}('
    for a in args:
        func += f'{a}, '
    for k, a in kwargs.items():
        func += f'{k}={a}, '
    func += ')'
    func = func.replace(', )', ')')
    print(f"Exception in {target}\n\n{func}\n\n{traceback}")
