import sys

if sys.version_info >= (3, 10):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata


VERSION = importlib_metadata.version("unasyncd")
