#!/usr/bin/env python3
# Initially proposed by Sebastian Ehlert (@awvwgk)
from os import environ, listdir, makedirs, walk
from os.path import join, isdir, exists
from sys import argv
from shutil import copy

build_dir = environ["MESON_BUILD_ROOT"]
if "MESON_INSTALL_DESTDIR_PREFIX" in environ:
    install_dir = environ["MESON_INSTALL_DESTDIR_PREFIX"]
else:
    install_dir = environ["MESON_INSTALL_PREFIX"]

include_dir = "modules"
module_dir = join(install_dir, include_dir)

modules = []
# finds $build_dir/**/*.mod and $build_dir/**/*.smod
for root, dirs, files in walk(build_dir):
    modules += [join(root, f) for f in files if f.endswith(".mod") or f.endswith(".smod")]

if not exists(module_dir):
    makedirs(module_dir)

for mod in modules:
    print("Installing", mod, "to", module_dir)
    copy(mod, module_dir)
