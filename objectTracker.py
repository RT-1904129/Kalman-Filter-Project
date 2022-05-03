# Importing All relevent library

import dataclasses
import cv2
import time
import imageio
import numpy as np
from tracker import Tracker 

images = []

# This function used to create Images
def CreateImages(width,height):
	size_of_image = (width, height, 3)
	images = np.ones(size_of_image,np.uint8)*255
	return images

def main():
    # Here we are loading the data which is avilabe in R,G,B formate
    data = np.array(np.load('Detections.npy'))[0:10,0:150,0:150]

    # Here we are creating Tracker objects
    tracker = Tracker(150, 30, 5)

    # List of all all Tracked Colors
    track_colors = [(126, 0, 255), (126, 0, 128),(124, 10, 255), (0,255, 125),
                    (245, 0, 0), (0, 245, 0), (0, 0, 245), (245, 245, 0),
                    (126, 126, 245), (246, 0, 255), (245, 127, 245)]

    for i in range(data.shape[1]):
        center = data[:,i,:]
        # Here we are creating frame
        frame = CreateImages(512,512)

        if (len(center) > 0):

            tracker.update(center)

            for j in range(len(tracker.tracks)):

                if(len(tracker.tracks[j].trace) > 1):

                    x = int(tracker.tracks[j].trace[-1][0,0])
                    y = int(tracker.tracks[j].trace[-1][0,1])

                    rectangle_length = (x-10,y-10)
                    rectangle_width = (x+10,y+10)
                    color = track_colors[j]
                    
                    
                    # it is used to create a rectangle box in image
                    # cv2.rectangle(image, start_point, end_point, color, thickness)
                    cv2.rectangle(frame,rectangle_length,rectangle_width,color,1)
                    
                    # it is used to write the comments on the rectangle frame
                    # cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
                    cv2.putText(frame,str(tracker.tracks[j].trackId), (x-9,y-10),0, 0.5, color,2)

                    for k in range(len(tracker.tracks[j].trace)):
                        x = int(tracker.tracks[j].trace[k][0,0])
                        y = int(tracker.tracks[j].trace[k][0,1])

                        # cv2.circle(image, center_coordinates, radius, color, thickness)
                        cv2.circle(frame,(x,y), 3, color,-1)

                    cv2.circle(frame,(x,y), 6, color,-1)
                cv2.circle(frame,(int(data[j,i,0]),int(data[j,i,1])), 6, (0,0,0),-1)

            # it is used to show frame which name is "Pictures" 
            cv2.imshow('Pictures ',frame)
            
            time.sleep(0.05)

            # It will destroy all windows if we press the key 3 or keyboard key s 
            if cv2.waitKey(3) & 0xFF == ord('s'):
                cv2.destroyAllWindows()
                break
            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

if __name__ == '__main__':
	main()