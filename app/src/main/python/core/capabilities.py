import platform
import shutil
import os

class PlatformCaps:
    """A simple class to hold detected platform capabilities."""
    def __init__(self):
        self.os = platform.system()
        self.arch = platform.machine()
        self.is_termux = 'TERMUX_VERSION' in os.environ
        self.has_cuda = self.os == "Linux" and shutil.which('nvcc') is not None
        self.has_metal = self.os == "Darwin" and self.arch == "arm64"
        self.gpu_type = "cuda" if self.has_cuda else "metal" if self.has_metal else "none"

    def __str__(self):
        return f"OS: {self.os}, Arch: {self.arch}, GPU: {self.gpu_type}, Termux: {self.is_termux}"

# Instantiate the capabilities class globally so it can be accessed by other functions.
CAPS = PlatformCaps()