Chuck's CUDA Tools (CCT)
------------------------

This repository is a small collection of CUDA wrappers and example code created
for my GTC 2016 talk entitled, 'S6283 GPU-Centric Thinking: Use Case Acceleration
of a DNA Sequencer Pipeline'. The latest code can be obtained from:

	https://github.com/chuckseberino/GTC16

Building
--------

CCT requires CMake (3.0+) to configure and CUDA 7.5. It has been tested with the
following OSes:

* Windows 7 (x64) with Visual Studio 2013 Update 5
* OSX El Capitan (10.11.3) with XCode 7.2.1
* Ubuntu 14.04 x64 with GCC 9.4.1

The code currently targets compute capability 3.0, 3.5, and 5.2, which should
cover most GPUs within the last few years.
