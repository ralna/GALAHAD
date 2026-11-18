#!/bin/bash
# Compile, but do not link, all of the GALAHAD Matlab interfaces in src/matlab
# against a stub of Matlab's fintrf.h, to catch errors in them without Matlab.
#
# usage: compile_matlab.sh <meson build directory containing libgalahad_double>
#
# The interfaces depend on each other, so the files are compiled repeatedly
# until no more succeed; any that still fail are then reported.

set -u
if [ $# -ne 1 ]; then
  echo "usage: $0 <meson build directory>" ; exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
galahad=$(cd "$here/../.." && pwd)
build=$(cd "$1" && pwd)
modules=$build/libgalahad_double.so.p
[ -d "$modules" ] || modules=$build/libgalahad_double.a.p
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
FC=${FC:-gfortran}

todo=$(ls "$galahad"/src/matlab/*.F90)
while true; do
  left=""
  for f in $todo; do
    name=$(basename "$f" .F90)
    if ! "$FC" -cpp -c -ffree-line-length-none -fallow-argument-mismatch \
           -I"$here" -I"$galahad/include" -I"$modules" -J"$work" -I"$work" \
           "$f" -o "$work/$name.o" > "$work/$name.log" 2>&1 ; then
      left="$left $f"
    fi
  done
  [ "$left" = "$todo" ] || [ -z "$left" ] && break
  todo=$left
done

if [ -z "$left" ]; then
  echo "all $(ls "$galahad"/src/matlab/*.F90 | wc -l) Matlab interface files compiled"
  exit 0
fi
for f in $left; do
  name=$(basename "$f" .F90)
  echo "==== $name failed to compile:"
  grep -E -B4 "Error" "$work/$name.log" | head -20
done
exit 1
