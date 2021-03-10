from threading import Thread, _active  # noqa
from traceback import format_exc
from typing import Callable, Optional, Any, Iterable, Mapping
from ctypes import pythonapi, py_object


def default_exc_callback(target: Callable[[Any], None], args: Iterable,
                         kwargs: Mapping[str, Any], traceback: str, e: Exception):
    """
    Default exception callback for MyThread. Pretty print the error and where happened
    """
    func = f'{target.__name__ if hasattr(target, "__name__") else target}('
    for a in args:
        func += f'{a}, '
    for k, a in kwargs.items():
        func += f'{k}={a}, '
    func += ')'
    func = func.replace(', )', ')')
    print('-' * 40)
    print(f"{type(e).__name__} in {target}\n\n{func}\n\n{traceback}")
    print('-' * 40)


class MyThread(Thread):

    def __init__(self, exc_callback=default_exc_callback, daemon: bool = True, *args, **kwargs):
        """
        Thread but handle exceptions with exc_callback and have get_id() and stop() functions

        :param exc_callback: function receiving target, args, kwargs, the traceback and Exception of any error
        :param daemon: stop thread when code finishes?
        :param args: "target" is needed
        """
        self._args = ()
        self._kwargs = {}
        super().__init__(daemon=daemon, *args, **kwargs)
        self._exc_callback = exc_callback

    def _target(self, *args, **kwargs):
        pass

    def run(self):
        try:
            self._target(*self._args, **self._kwargs)
        except Exception as e:
            self._exc_callback(self._target, self._args, self._kwargs, format_exc(), e)

    def get_id(self) -> Optional[int]:
        """Return the Thread ID"""
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for thread_id, thread in _active.items():
            if thread is self:
                return thread_id

    def stop(self):
        """Try to raise an exception in running thread to stop it"""
        thread_id = self.get_id()
        res = pythonapi.PyThreadState_SetAsyncExc(thread_id, py_object(SystemExit))
        if res > 1:
            pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            self._exc_callback(self._target, self._args, self._kwargs,
                               ".stop() call to MyThread failed", Exception())
