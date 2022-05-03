# Importing the required libraries & modules.
import numpy as np 
from collections import deque
from kalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment

# class that represents each individual object being tracked by the tracker.
class TrackItem(object):
	def __init__(self, detectionCoord, trackId):
		super(TrackItem, self).__init__()
		
		# For every "TrackItem" object, we make a KalmanFilter.
		self.kalFil = KalmanFilter()
		self.kalFil.predict()
		self.kalFil.update(np.matrix(detectionCoord).reshape(2,1))
		self.prediction = detectionCoord.reshape(1,2)

		# Initialising the parameters.
		self.trace = deque(maxlen=20)
		self.trackId = trackId
		self.framesSkipped = 0

	def predict(self,detectionCoord):

		# Predict the next state, reshape it & store it.
		self.prediction = np.array(self.kalFil.predict()).reshape(1,2)

		# Reshaping the current detection to update states.
		currMeasurement = np.matrix(detectionCoord).reshape(2,1)

		# Updating the Kalman Filter states based on the current measurement.
		self.kalFil.update(currMeasurement)


class Tracker(object):
	def __init__(self, thresholdDist, maxFrameSkipped, maxTraceLength):
		super(Tracker, self).__init__()

		# Storing the parameters in the attributes of the class.
		self.thresholdDist = thresholdDist
		self.maxFrameSkipped = maxFrameSkipped
		self.maxTraceLength = maxTraceLength
		self.trackId = 0
		self.tracks = []

	def update(self, detectionCoord):

		# If there are no tracks, check all the detections in the current frame and create a tracks object for each of them.
		# This if will be satisfied only for the first frame in the inout data.
		if len(self.tracks) == 0:
			for i in range(detectionCoord.shape[0]):
				track = TrackItem(detectionCoord[i], self.trackId)
				self.trackId +=1
				self.tracks.append(track)


		N = len(self.tracks)
		costArr = []

		# Iterating over each track & computing a cost array which has the the distances for each pair of ( Detection, Estimation/Prediction ) 
		for i in range(N):
			diff = np.linalg.norm(self.tracks[i].prediction - detectionCoord.reshape(-1,2), axis=1)
			costArr.append(diff*0.1)

		# Converting list to numpy array.
		costArr = np.array(costArr)

		# Finding the complete assignment ( Mapping between detection & prediction ) for all objects such that the cost is minimum possible.
		# It internally uses algo similiar to Hungarian algorithm.
		row, col = linear_sum_assignment(costArr)

		# To store the assignment of each of the N objects tracked.
		assignList = [-11]*N
		for i in range(len(row)):
			assignList[row[i]] = col[i]


		unAssignedTracks = []
		# For each object, If it is assigned, We check the cost it takes to assign & unassign if it exceeds the threshold.
		for i in range(N):
			if assignList[i] != -11:
				if (costArr[i][assignList[i]] > self.thresholdDist):
					assignList[i] = -11
					unAssignedTracks.append(i)
				else:
					self.tracks[i].framesSkipped += 1

		# For each of the tracked object, If it exceeds the max_frame, we append it to array to delete later.
		toDeleteTracks = []
		for i in range(len(self.tracks)):
			if self.tracks[i].framesSkipped > self.maxFrameSkipped :
				toDeleteTracks.append(i)

		# If there are any elements to be deleted, delete the tracker object & also its assignment.		
		if len(toDeleteTracks) > 0:
			for i in toDeleteTracks:
				del self.tracks[i]
				del assignList[i]

		# If any object has no assignment to it, create a new track object & append it to tracks.
		for i in range(len(detectionCoord)):
			if i not in assignList:
				track = TrackItem(detectionCoord[i], self.trackId)
				self.trackId +=1
				self.tracks.append(track)

		# At last, If any object is assigned, We are reinitialising the frame count & starting prediciton 
		# and then we are storing the next predicted state to deque. 
		for i in range(len(assignList)):
			if(assignList[i] != -1):
				self.tracks[i].framesSkipped = 0
				self.tracks[i].predict(detectionCoord[assignList[i]])

			self.tracks[i].trace.append(self.tracks[i].prediction)
