"""charnet — Time-Evolved Character Interaction Network from Transcripts & Scene Detection."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("charnet")
except PackageNotFoundError:
    __version__ = "0.1.0"
