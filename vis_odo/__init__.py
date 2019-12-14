"""
Visual Odometry module used to determine position and attitude from camera
images over time.
"""

import os.path
from threading import Thread, Lock
import time

import cv2 as cv
from cv2 import ximgproc
import numpy as np
import hjson

from sparam import load_params

from .pose import Pose
from .stereo_image_source import StereoImageSource

class VisOdo:
    """
    Visual Odometry Class, implements the Visual Odometry system for the rover.

        usage
        -----
            # Create new instance of the class
            vis_odo = VisOdo("path/to/params/file.hjson")

            # Create a new image source, in this case the dataset 'devon'
            stereo_image_source = StereoImageSource.from_dataset("devon")

            # Start the processing, passing in the image source
            vis_odo.start(stereo_image_source)

            # Every 5 seconds get the delta pose estimate, can be used to
            # propagate the overall pose of the vehicle
            while True:
                delta_pose = vis_odo.get_delta_pose_estimate()

                ...
    """

    def __init__(self):
        # pylint: disable=no-member
        """
        Initialise the system by loading the necessary parameters.
        """

        # Get the path to this file, since relative imports are messed up and
        # are relative to the executing script's path, not the path of this
        # file
        vis_odo_dir = os.path.dirname(os.path.realpath(__file__))

        # Load the parameter file using sparam
        with open(os.path.join(vis_odo_dir, '../params/vis_odo.hjson')) as f:
            self.params = hjson.loads(f.read())
        with open(os.path.join(vis_odo_dir, '../params', 
                               self.params["stereo_bench_params"])) as f:
            self.stereo_params = hjson.loads(f.read())

        # Initialise opencv modules
        self.orb = cv.ORB_create()
        self.bf_matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.stereo_left = cv.StereoBM_create(
            self.params["stereo_max_disparity"],
            self.params["stereo_block_size"])
        self.wsl_filter = ximgproc.createDisparityWLSFilter(self.stereo_left)
        self.stereo_right = ximgproc.createRightMatcher(self.stereo_left)

        # Set opencv parameters
        self.wsl_filter.setLambda(self.params["stereo_lambda"])
        self.wsl_filter.setSigmaColor(self.params["stereo_sigma"])

        # Define internal data
        self.sim_time_s = None
        self.image_source = None
        self.current_frame_accessed = False
        self.current_frame = None
        self.last_frame = None
        self.latest_delta_pose_est = None
        self.status_rpt = {}
        self.lock = Lock()


    def start(self, image_source):
        """
        Start the image processing loop, which shall be run in a seperate
        thread.

        TODO: This could need to be spun off as a separate process/implemented
        in something like Rust/C++ due to GIL :(
        """
        # Assign the image source
        self.image_source = image_source

        # Spin off the loop thread
        Thread(target=self._loop).start()


    def get_current_frame(self):
        """
        Get the current frame provided by the image source.

        returns
        -------
        last_frame: If a new frame is available. Once the frame is returned it
            shall be cleared internally so future calls to this function will
            return None until a new frame is available.
        None: If no new frame is available
        """

        # Get the lock
        if self.lock.acquire():
            # Make copy of the frame
            current_frame = self.current_frame

            # If the current frame has already been accessed return None,
            # otherwise set the accessed variable to true
            if self.current_frame_accessed:
                current_frame = None
            else:
                self.current_frame_accessed = True

            # Release the lock
            self.lock.release()

            return current_frame

        # If lock failed
        raise Exception("Failed to get lock in VisOdo.get_current_frame")


    def get_status_rpt(self):
        """
        Returns the status report of the processing.
        """

        # Get the lock
        if self.lock.acquire():

            # Make copy of the report
            status_rpt = self.status_rpt

            self.lock.release()

            return status_rpt

        # If lock failed
        raise Exception("Failed to get lock in VisOdo.get_status_rpt")


    def get_delta_pose_est(self):
        """
        Get the estimated delta pose (change in position/attitude) since the
        last frame.

        returns
        -------
        delta_pose_est: If a new pose estimate is available. Once the pose
            estimate is returned it shall be cleared internally so future calls
            to this function will return None until a new pose estimate is
            available.
        None: If no new pose estimate is available
        """

        # Get the lock
        if self.lock.acquire():

            # Make a copy of the pose
            delta_pose = self.latest_delta_pose_est

            # If there is a new pose then clear the stored pose
            if delta_pose is not None:
                self.latest_delta_pose_est = None

            # Release lock
            self.lock.release()

            return delta_pose

        # If lock failed
        raise Exception("Failed to get lock in VisOdo.get_delta_pose_est")


    def set_sim_time(self, sim_time_s):
        """
        Update the internal simulation time of the VisOdo loop.
        """

        # Attempt to acquire the lock (blocking)
        if self.lock.acquire():

            # Update internal sim time
            self.sim_time_s = sim_time_s
            self.lock.release()

        # If lock failed
        else:
            raise Exception("Failed to get lock in VisOdo.set_sim_time")


    def _loop(self):
        """
        Main processing loop for the Visual Odometry system
        """

        # Copy of last sim_time to detect system steps
        last_sim_time_s = self.sim_time_s

        # Activate the image source
        self.image_source.activate()

        # Keep running this loop until the source runs out of images
        while self.image_source.active:

            # Get the current sim_time
            if self.lock.acquire():
                sim_time_s = self.sim_time_s
                self.lock.release()

                # If the simulation time has not been initialised
                if sim_time_s is None:
                    continue

                # If this is the first sim_time change after an init
                if last_sim_time_s is None:
                    last_sim_time_s = sim_time_s - 10

                # If the sim_time has been updated
                if sim_time_s > last_sim_time_s + 0.001:

                    # Step the processing
                    status_rpt = self._step()

                    # Lock to update status report
                    if self.lock.acquire():
                        self.status_rpt.update(status_rpt)
                        self.lock.release()

                    else:
                        raise Exception(
                            "Failed to get lock while updating status report "
                            "in main loop")

            # If the lock couldn't be acquired
            else:
                raise Exception(
                    "Failed to get lock when accessing VisOdo.sim_time_s "
                    "in main loop")

        # Set in the status report that the source has gone inactive
        if self.lock.acquire():
            self.status_rpt["image_source_inactive"] = {
                "is_critical": True,
                "message": "Image source is inactive, either the camera source"
                           " has stopped or the end of the dataset has been "
                           "reached."}
            self.lock.release()

        # If the lock was unavailable return an error
        else:
            raise Exception(
                "Failed to get lock while formatting status report on image "
                "source deactivation")



    def _step(self):
        """
        Step the visual odometry system. Returns a status report indicating the
        success of the operations.
        """

        # Status report which will be updated with the module report on return
        status_rpt = {}

        # Get the latest frame from the image source
        if self.lock.acquire():
            frame = self.image_source.get_frame(self.sim_time_s)

            # If we got a new frame shift the current frame into the last frame
            if frame is not None:
                print(f"New frame acquired at {self.sim_time_s:.2f} s")
                self.last_frame = self.current_frame
                self.current_frame = None

            self.lock.release()

            # If there's no new frame no need to do any processing
            if frame is None:
                return status_rpt

        else:
            raise Exception(
                "Failed to get lock while acquiring latest frame in "
                "VisOdo._step")

        # Record the start time of this processing
        time_start = time.time()

        # ---- FEATURE DETECTION ----

        # Detect keypoints and compute descriptors for the keypoints
        keypoints_left, descriptors_left = self.orb.detectAndCompute(
            frame.img_left, None)
        keypoints_right, descriptors_right = self.orb.detectAndCompute(
            frame.img_right, None)

        if self.params["plot_keypoints"]:
            # Draw the keypoints onto two new images and save them into the a
            # member of the imgs_processed dictionary
            frame.imgs_processed["left_with_keypoints"] = cv.drawKeypoints(
                frame.img_left, keypoints_left, color=(0, 255, 0), flags=0,
                outImage=None)
            frame.imgs_processed["right_with_keypoints"] = cv.drawKeypoints(
                frame.img_right, keypoints_right, color=(0, 255, 0), flags=0,
                outImage=None)

        # Here we save the data we gained during the processing for use in the
        # next step
        frame.data["keypoints_left"] = keypoints_left
        frame.data["descriptors_left"] = descriptors_left
        frame.data["keypoints_right"] = keypoints_right
        frame.data["descriptors_right"] = descriptors_right

        # ---- DISPARITY MAP GENERATION -----

        # Convert images to greyscale for disaprity calculations
        img_left_grey = cv.cvtColor(frame.img_left, cv.COLOR_BGR2GRAY)
        img_right_grey = cv.cvtColor(frame.img_right, cv.COLOR_BGR2GRAY)

        # Calculate disparity maps
        disparity_left = self.stereo_left.compute(
            img_left_grey, img_right_grey)
        disparity_right = self.stereo_right.compute(
            img_right_grey, img_left_grey)

        # Filter disparity maps
        frame.data["disparity_map"] = self.wsl_filter.filter(
            disparity_left, frame.img_left, None, disparity_right)

        # Calculate depth maps
        frame.data["depth_map"] = np.divide(
            self.stereo_params["stereo_baseline_m"]
            * self.stereo_params["focal_length_m"],
            frame.data["disparity_map"])

        # Plot the disparity map
        if self.params["plot_depth_map"]:
            # Need to normalise the depth map so we can display it in u8,
            # normalize based on the maximum depth map distance
            # TODO: this doesn't actually work, end up with very small depths
            depth_norm = np.vectorize(
                lambda x: x * 255 / self.params["depth_map_max_dist_m"])(
                    frame.data["depth_map"])
            print(f"depth: {depth_norm[100][200]}, {frame.data['depth_map'][100][200]}")
            frame.imgs_processed["depth_map"] = cv.cvtColor(
                np.uint8(depth_norm), cv.COLOR_GRAY2BGR)

        # From this point on if the last_frame is None no more processing can
        # be done so we skip to the end
        if self.last_frame is not None:

            # ---- FEATURE MATCHING ----
            frame = self._match_features(frame)


        # End time
        time_end = time.time()
        print(f"VisOdo compute time: {(time_end - time_start):.6f} s")

        # Set the pose est and current frame
        if self.lock.acquire():
            self.current_frame = frame
            self.current_frame_accessed = False

            # TODO: remove, for debug only
            self.latest_delta_pose_est = Pose()

            self.lock.release()
        else:
            raise Exception(
                "Failed to get lock when setting last frame in VisOdo._step")

        return status_rpt

    def _match_features(self, current_frame):
        """
        Match features between the current frame and the last frame
        """

        # Compute matches between this frame and the last frame
        matches_left = self.bf_matcher.match(
            self.last_frame.data["descriptors_left"],
            current_frame.data["descriptors_left"])
        matches_right = self.bf_matcher.match(
            self.last_frame.data["descriptors_right"],
            current_frame.data["descriptors_right"])

        # Local variables so we don't have to do
        # current_frame.data["keypoints_"] all the time
        keypoints_left = current_frame.data["keypoints_left"]
        keypoints_right = current_frame.data["keypoints_right"]

        # Sort the matches based on distance
        matches_left = sorted(matches_left, key=lambda x: x.distance)
        matches_right = sorted(matches_right, key=lambda x: x.distance)

        # Create images where a number of closest matches are in green,
        # weak matches in blue, and ones with no match in red
        idxs_left_strong = [
            m.trainIdx for m in matches_left[0:self.params["num_strong_points"]]]
        idxs_right_strong = [
            m.trainIdx for m in matches_right[0:self.params["num_strong_points"]]]

        idxs_left_weak = [
            m.trainIdx for m in matches_left[self.params["num_strong_points"]:-1]]
        idxs_right_weak = [
            m.trainIdx for m in matches_right[self.params["num_strong_points"]:-1]]

        keypoints_left_no_match = [keypoints_left[j] for j in
            [i for i in range(len(keypoints_left))
                if (i not in idxs_left_strong and i not in idxs_left_weak)]]
        keypoints_right_no_match = [keypoints_right[j] for j in
            [i for i in range(len(keypoints_right))
                if (i not in idxs_right_strong and i not in idxs_right_weak)]]

        # Plot no match keypoints
        match_strengh_img_left = cv.drawKeypoints(
            current_frame.img_left, keypoints_left_no_match, color=(255, 0, 0),
            flags=0, outImage=None)
        match_strength_img_right = cv.drawKeypoints(
            current_frame.img_right, keypoints_right_no_match, color=(255, 0, 0),
            flags=0, outImage=None)

        # Plot weak keypoints
        match_strengh_img_left = cv.drawKeypoints(
            match_strengh_img_left,
            [keypoints_left[i] for i in idxs_left_weak], color=(0, 0, 255),
            flags=0, outImage=None)
        match_strength_img_right = cv.drawKeypoints(
            match_strength_img_right,
            [keypoints_right[i] for i in idxs_right_weak],
            color=(0, 0, 255), flags=0, outImage=None)

        # Plot the strong keypoints
        current_frame.imgs_processed["left_match_strength"] = cv.drawKeypoints(
            match_strengh_img_left,
            [keypoints_left[i] for i in idxs_left_strong],
            color=(0, 255, 0), flags=0, outImage=None)
        current_frame.imgs_processed["right_match_strength"] = cv.drawKeypoints(
            match_strength_img_right, 
            [keypoints_right[i] for i in idxs_right_strong],
            color=(0, 255, 0), flags=0, outImage=None)

        # Assign current frame data for the strong matches
        current_frame.data["matches_left_strong"] = matches_left[
            0:self.params["num_strong_points"]]
        current_frame.data["matches_right_strong"] = matches_right[
            0:self.params["num_strong_points"]]

        return current_frame
