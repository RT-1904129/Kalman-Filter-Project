# Importing the required libraries & modules.
import cv2
import time
import numpy as np
from trackHelp import Tracker

# Loading the coordinatess ( x, y coordinates ) of the 10 objects in around 300 frames from the .npy format.
data = np.array(np.load("detectedData.npy"))[0:10, :, :]

# Creating one Tracker object with dist_threshold = 150, max_frame_skipped = 30, max_trace_length = 5
tracker = Tracker(150, 30, 5)

# List of colors used for tracking.
colors = [(129, 0, 255), (129, 0, 128), (124, 10, 255), (0, 255, 125),
                (245, 0, 0), (0, 245, 0), (0, 0, 245), (245, 245, 0),
                (129, 129, 245), (246, 0, 255), (245, 127, 245)]

for frame_i in range(data.shape[1]):

    coordinates = data[:, frame_i, :]

    # Creating a window to show output tracking using opencv.
    window = np.ones((512, 512, 3), np.uint8)*255

    if (len(coordinates) > 0):
        tracker.update(coordinates)
        for obj_idx in range(len(tracker.tracks)):

            # If there are atleast 2 elements in its deque
            if (len(tracker.tracks[obj_idx].trace) > 1):

                # Grab the coordinates, define the dimensions of rectangle to enclose that object.
                x = int(tracker.tracks[obj_idx].trace[-1][0, 0])
                y = int(tracker.tracks[obj_idx].trace[-1][0, 1])
                rectangleLength = (x-10, y-10)
                rectangleWidth = (x+10, y+10)

                # Choosing the color.
                color = colors[obj_idx]

                # Creating a rectangle box enclosing the object in the window using cv2.rectangle(image, start_point, end_point, color, thickness)
                cv2.rectangle(window, rectangleLength,
                              rectangleWidth, color, 1)

                # Write the id number onto the  window using cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
                cv2.putText(window, str(
                    tracker.tracks[obj_idx].trackId), (x-9, y-10), 0, 0.5, color, 2)

                # For each element in the deque, grab the coordinates & draw small circles which act as tail along the path for the object.
                for k in range(len(tracker.tracks[obj_idx].trace)):
                    x = int(tracker.tracks[obj_idx].trace[k][0, 0])
                    y = int(tracker.tracks[obj_idx].trace[k][0, 1])

                    # Drawing small circles with color chosen using cv2.circle(image, coordinates_coordinates, radius, color, thickness)
                    cv2.circle(window, (x, y), 3, color, -1)

                cv2.circle(window, (x, y), 6, color, -1)

            cv2.circle(window, (int(data[obj_idx, frame_i, 0]), int(
                data[obj_idx, frame_i, 1])), 6, (0, 0, 0), -1)

        # Showing the entire window made above to the user's screen.
        cv2.imshow('Object Tracking using Kalman Filter!', window)

        # Setting some delay between the succesive frames of the input.
        time.sleep(0.05)

        # It closes the entire window when 'x' is pressed.
        if cv2.waitKey(5) & 0xFF == ord('x'):
            cv2.destroyAllWindows()
            break
