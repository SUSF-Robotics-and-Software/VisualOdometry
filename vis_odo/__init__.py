"""
Visual Odometry module used to determine position and attitude from camera
images over time.
"""

import os.path
from threading import Thread, Lock

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
        """
        Initialise the system by loading the necessary parameters.
        """

        # Get the path to this file, since relative imports are messed up and
        # are relative to the executing script's path, not the path of this
        # file
        loco_ctrl_dir = os.path.dirname(os.path.realpath(__file__))

        # Load the parameter file using sparam
        self.params = load_params.load_params_from_hjson(
            os.path.join(loco_ctrl_dir, '../params/vis_odo.hjson'))

        # Define internal data
        self.sim_time_s = None
        self.image_source = None
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


    def get_last_frame(self):
        """
        Get the last frame provided by the image source.

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
            last_frame = self.last_frame

            # Clear the frame if it exists
            if last_frame is not None:
                self.last_frame = None

            # Release the lock
            self.lock.release()

            return last_frame

        # If lock failed
        raise Exception("Failed to get lock in VisOdo.get_last_frame")


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

            if frame is not None:
                print(f"New frame acquired at {self.sim_time_s:.2f} s")

            self.lock.release()
        else:
            raise Exception(
                "Failed to get lock while acquiring latest frame in "
                "VisOdo._step")

        # TODO: processing


        # Set the pose est and last frame
        if self.lock.acquire():
            # If the current value of last frame is not none (has not yet been)
            # cleared by a call to get_last_frame, then we should only
            # overwrite it if a new image is available
            if self.last_frame is not None and frame is not None:
                self.last_frame = frame
            elif self.last_frame is None:
                self.last_frame = frame

            # TODO: remove, for debug only
            self.latest_delta_pose_est = Pose()

            self.lock.release()
        else:
            raise Exception(
                "Failed to get lock when setting last frame in VisOdo._step")

        return status_rpt
