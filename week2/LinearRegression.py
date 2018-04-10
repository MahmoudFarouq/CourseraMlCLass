import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time as t

class LinearRegression:

	def fit(self, X, y, alpha=0.01, maxIter=1500):
		# X: np.Array() -> matrix of size (m*n) -> the features matrix
		# y: np.Array() -> vector of size (m*1) -> the labels vector
		# alpha: int -> the learningRate of the algorithm
		# maxIter: int -> maximum number of iteration the gradient dexcent can do before convergence

		# m -> number of examples
		# n -> number of features
		m, n = X.shape

		# 1) adding the bias term
		bias = np.ones((m, 1))
		X = np.hstack((bias, X))

		# 2) initialize theta to zeroes
		self.theta = np.zeros(n+1)

		# 3) applying the gradient descent
		self.GradientDescent(X, y, alpha, maxIter)

		# 4) plotting the data on a 3D plot
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(X[:, 1], X[:, 2], y, zdir='z', s=20, c=None, depthshade=True)
		ax.plot(X[:, 1], X[:, 2], [self.h(x) for x in X])
		plt.show()

	def GradientDescent(self, X, y, alpha, maxIter):
		n = X.shape[1]

		# the main loop, iterates till maxIter is zero
		while maxIter > 0:

			# copy of the theta vector, { it's because we need to update contiguesly }
			temp_theta = self.theta.copy()

			# for every theta in the theta vector
			for j in range(n):

				# tmp_theta = old_theta - alpha * d(J(theta)) / d(theta(j))
				temp_theta[j] = self.theta[j] - alpha * self.computeCostDiff(X, y, j)

			# make the old theta = the new theta
			self.theta = temp_theta

			maxIter -= 1

	# computes this term { d(J(theta)) / d(theta(j)) }
	def computeCostDiff(self, X, y, j):
		m = X.shape[0]
		diff = 0
		for i in range(m):
			diff += (self.h(X[i])- y[i]) * X[i][j]
		return (1/m) * diff 

	# the hyposis
	def h(self, x):
		return self.theta.T.dot(x)

	# the cost function
	def J(self, X, y):
		m = X.shape[0]
		sum = 0
		for i in range(m):
			sum += ( self.h(X[i]) - y[i] )**2
		return sum * (1/(2*m))

	# predict function
	def predict(self, x):
		return self.h(x)


if __name__ == '__main__':
	data = pd.read_csv('data.txt')
	X = data.values
	y = np.random.randint(50, size=(96, 1))
	clf = LinearRegression()
	clf.fit(X, y)