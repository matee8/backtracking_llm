"""Provides private, internal UI utility classes for the project's scripts.

NOTE: The contents of this module are not part of the public library API and are
not intended for use by end-users of the library.
"""

import time
from threading import Thread


class Spinner:
    """A simple spinner context manager for long-running operations.

    Usage:
        with Spinner("Loading..."):
            time.sleep(5)
    """

    def __init__(self, message: str, delay: float = 0.1) -> None:
        """Initializes the Spinner.

        Args:
            message: The message to display next to the spinner.
            delay: The delay in seconds between each frame of the animation.
        """
        self.spinner_thread = Thread(target=self._spin)
        self.message = message
        self.delay = delay
        self.busy = False

    def _spin(self):
        while self.busy:
            for char in '|/-\\':
                print(f'\r{self.message} {char}', end='', flush=True)
                time.sleep(self.delay)

    def __enter__(self):
        """Starts the spinner."""
        self.busy = True
        self.spinner_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stops the spinner and clears the line."""
        self.busy = False
        self.spinner_thread.join()
        print(f"\r{' ' * (len(self.message) + 2)}\r", end='', flush=True)
