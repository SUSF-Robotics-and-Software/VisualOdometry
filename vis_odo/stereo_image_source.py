from enum import Enum
import numpy as np
import cv2 as cv
import dateutil.parser
import hjson
import os.path
import glob
import re

class StereoFrame:
    """
    Class storing a pair of images, a simulation time, and the pose of the 
    left camera frame in the world sim frame.
    """
    img_left = None
    img_right = None
    sim_time_s = None
    lc_pose_ws = None

    def __init__(self, sim_time_s):
        self.sim_time_s = sim_time_s

class StereoImageSource:
    """
    Class providing a source for images for the visodo processing.
    """

    class SourceType(Enum):
        """
        Enum which specifies where this source is going to get it's images from.

        NONE - Source not yet initialised, no images will be produced
        DATASET - Images will come from a dataset and will be available at 
            specific times
        
        TODO: Add camera streams
        """
        NONE = 0
        DATASET = 1

    def __init__(self):
        self.last_accessed_index = False
        self.source_type = StereoImageSource.SourceType.NONE

    @classmethod
    def from_dataset(cls, source_param_file_path):
        """
        Setup a dataset as the source of the images
        """
        src = cls()

        src.source_type = StereoImageSource.SourceType.DATASET
        src.source_param_file_path = source_param_file_path

        # Load the parameters
        with open(source_param_file_path) as f_params:
            src.params = hjson.loads(f_params.read())

        # Unescape the regex pattern
        img_regex_pattern = os.path.join(
            src.params["image_dir_path"], 
            src.params["image_camera_index_regex"])\
                .encode().decode('unicode_escape')

        # Compile the image name regex
        img_regex = re.compile(img_regex_pattern)

        # Get the list of images in the data directory
        raw_image_list = glob.glob(os.path.join(
            src.params["image_dir_path"], src.params["image_glob"]))

        # Parse the raw list into a dictionary
        img_dict = {}
        for img in raw_image_list:
            m = img_regex.match(img)

            # If the match is successful check if this is the left or right
            # image and then add it to the dictionary
            if m:
                # If an image with this index has not been added
                if m.group(2) not in img_dict.keys():
                    img_dict[m.group(2)] = {}

                if m.group(1) == src.params["camera_name_left"]:
                    img_dict[m.group(2)]['left'] = img
                elif m.group(1) == src.params["camera_name_right"]:
                    img_dict[m.group(2)]['right'] = img

        # Load the timeline file
        with open(src.params["timeline_file_path"]) as f_tl:
            line = f_tl.readline()

            init_time = dateutil.parser.parse(line.split(' ')[1])

            src.timeline = list()

            while line:
                line_parts = line.split(' ')

                # If the index matches one of the keys in the image dictionary
                if line_parts[0] in img_dict.keys():

                    # Append to the timeline the sim time and image dictionary 
                    # entry
                    sim_time_s = ((dateutil.parser.parse(line_parts[1])
                         - init_time).total_seconds())

                    src.timeline.append(
                        (sim_time_s, line_parts[0], img_dict[line_parts[0]]))

                line = f_tl.readline()

        # Load positional data
        with open(src.params["position_data_file_path"]) as f_pos:
            line = f_pos.readline()

            # TODO: Load position timeline correctly

            #while line:
            #    line_parts = line.split(' ')

        # Sort the timeline
        src.timeline.sort(key = lambda tup: tup[0] )

        return src

    def get_frame(self, sim_time_s):
        """
        Get the next pending frame based on the current sim time
        """

        # Iterating through the reversed timeline we can easily find the
        # most evolved timeline item which has a trigger time lower than 
        # the current sim_time.
        (frame_sim_time_s, frame_index, img_dict) = \
            next(((t, i, d) for (t, i, d) in reversed(self.timeline) \
                if t <= sim_time_s), (None, None, None))

        # Check that both the left and right images are available for the
        # indexed time
        if "left" not in img_dict.keys() or "right" not in img_dict.keys():
            # TODO: add low level "cannot find both images" log
            return None
        
        # If this found frame is the same as the previous frame
        elif self.last_accessed_index == frame_index:
            return None

        # Otherwise return the new frame
        else:
            # Begin building the Frame object
            frame = StereoFrame(frame_sim_time_s)

            frame.img_left = cv.imread(img_dict["left"])
            frame.img_right = cv.imread(img_dict["right"])

            # Set that this most recent frame has been accessed
            self.last_accessed_index = frame_index

            return frame

    