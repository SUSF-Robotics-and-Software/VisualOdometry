# VisualOdometry

This module provides Visual Odometry (visodo) for the Olympus rover project.

## Dependencies

This module requires OpenCV to be installed on your system. The module is 
currently under development in Python so installing OpenCV is just

```
pip install opencv-python
```

(*Note*: we may include the `opencv-contrib-python` package at some point in 
the future so worth installing that too.)

This should grab things like numpy as well which are required for OpenCV.

You will also need `hjson` for parsing the parameters files.

## Getting Datasets

Obviously we need some data to be able to test whether or not the system is 
working. Currently this takes the form of datasets, stored in the `datasets`
directory. Note that all the actual data is ignored by git because we don't
want to commit too much data into this directory.

See `datasets/readme.md` for information on getting them all setup.