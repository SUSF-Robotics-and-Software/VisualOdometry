import vis_odo
import numpy as np
import cv2 as cv
import math
from datetime import datetime

def main():

    # Test name, dataset, and time log
    print("---- VISODO TEST ----")
    print("  Devon Island Dataset")
    print(f"  {datetime.now()}")

    # Setup the image source
    print(
        "Loading image source from the `params/image_source_devon.hjson`"
        " parameter file")
    image_source = vis_odo.StereoImageSource.from_dataset(
        "params/image_source_devon.hjson")
    
    print(f"Loaded timeline of {len(image_source.timeline)} frames")

    # Setup simple simulation step
    sim_time_s = 0
    time_step_s = 0.1

    # Main sim loop
    while sim_time_s < 100:
        
        # Get the new frame
        frame = image_source.get_frame(sim_time_s)   

        if frame:
            print(
                f"{sim_time_s:.2f} s: got images for time "
                f"{frame.sim_time_s:.2f} s")

            # Concat images
            mosaic = np.hstack((frame.img_left, frame.img_right))

            # Add timestamp in bottom right
            mosaic_shape = np.shape(mosaic)
            text_pos = (mosaic_shape[1] - 100, mosaic_shape[0] - 20)
            #cv.rectangle(mosaic, text_pos, mosaic_shape, (0, 0, 0))
            cv.putText(
                mosaic, f"t = {frame.sim_time_s:.2f} s", text_pos, 
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness=1, 
                lineType=cv.LINE_AA)

            # Rescale the mosaic to give a good view
            mosaic_scaled = cv.resize(
                mosaic, (math.floor(mosaic_shape[1] * 1.5), 
                math.floor(mosaic_shape[0] * 1.5)))

            # Display the images
            cv.imshow(f"Devon Island Stereo Images", mosaic_scaled)
            cv.waitKey(math.floor(1000 * time_step_s))

        else:
            # No new frame, keep the old image and say we didn't get something 
            # new
            print(f"{sim_time_s:.2f} s: No new frame available yet")

        # Update sim time
        sim_time_s += time_step_s
        
    # Print exit message
    print("End of simulation run")

if __name__ == "__main__":
    main()