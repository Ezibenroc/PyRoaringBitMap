"""
Helper script to download a specific release amalgamation file for CRoaring.

Usage: python download_amalgamation.py <croaring_release_version, like v0.6.0>

The version needs to be the specific release tag on github.

"""
import os
import sys
from urllib.request import urlretrieve

version = sys.argv[1]

release = f"https://github.com/RoaringBitmap/CRoaring/releases/download/{version}/"

print(f"Downloading version {version} of the croaring amalgamation")

files = ["roaring.c", "roaring.h"]

for file in files:
    r = urlretrieve(release + file, os.path.join("pyroaring", file))

with open(os.path.join("pyroaring", "croaring_version.pxi"), "w") as f:
    f.write(f"__croaring_version__ = \"{version}\"")
