class Pose:
    """
    Class encapsulating the pose of one frame within another. A pose can be 
    thought of as both a position and attitude. Internally the position is 
    represented as a cartesian right handed vector in units of meters, while 
    the attitude is represented as a unit quaternion with q = [q_x q_y q_z q_w].

    Internally the Euler sequence 3-2-1 is used for converting Euler angles to
    quaternions.

        usage
        -----
            # Create a new pose of the body frame in the base frame and set it
            # to be at y = 1, x = z = 0, with a yaw of 0.5 radians.
            body_init_pose_base = Pose.from_pos_angles([0, 1, 0], [0, 0, 0.5])

            # Create a new pose which is based on some translation and rotation
            # of the original pose of the vehicle.
            body_next_pose_base = body_init_pose_base.
                transform([1, 0, 0], [0.5, 0, 0])
    """

    def __init__(self):
        self.pos_m = [0, 0, 0]
        self.att_q = [0, 0, 0, 1]

    @classmethod
    def from_pos_angles(cls, pos_m, angles_rad):
        """
        Create a new pose from the given position and angles. Position is a
        cartesian right handed vector in units of meters and the angles is a
        set of [roll, pitch, yaw] in units of radians.

            usage
            -----
                # Create a new pose of the body frame in the base frame and 
                # set it to be at y = 1, x = z = 0, with a yaw of 0.5 radians.
                body_init_pose_base = Pose.
                    from_pos_angles([0, 1, 0], [0, 0, 0.5])
        """
        pass

    def transform(self, translation_m, rotation_rad):
        pass