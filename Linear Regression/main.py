import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Reshape data -> sklearn need vertical format
time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(-1,1)
scores = np.array([56, 83, 47, 93, 47, 82, 45, 78, 55, 67, 57, 4, 12]).reshape(-1,1)

model = LinearRegression()
model.fit(time_studied, scores)


# A scatter plot of y vs. x with varying marker size and/or color.
plt.scatter(time_studied, scores)
plt.ylim(0,100)
plt.show()


# plot with linear regression line
# linspace parameters:
# (1) start - The starting value of the sequence.
# (2) stop - The end value of the sequence, unless endpoint is set to False.
# (3) start - Number of samples to generate. Default is 50.

plt.scatter(time_studied, scores)
plt.plot(np.linspace(0,70,100).reshape(-1,1), model.predict(np.linspace(0,70,100).reshape(-1,1)), 'r')
plt.ylim(0,100)
plt.show()

# score if you study for 56 hours
print("Score if you study for 56 hours: ", model.predict(np.array([56]).reshape(-1,1)))


# we want to train and score data using skleran.model_selection
# in example: 70% per training and 30% per test

time_studied2 = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(-1,1)
scores2 = np.array([56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57, 4, 89]).reshape(-1,1)
time_train, time_test, score_train, score_test = train_test_split(time_studied2, scores2, test_size=0.3)


model2 = LinearRegression()
model2.fit(time_train, score_train)

print(model2.score(time_test, score_test) * 100, "%")

plt.scatter(time_train, score_train)
plt.plot(np.linspace(0,70,100).reshape(-1,1), model2.predict(np.linspace(0,70,100).reshape(-1,1)), 'r')
plt.show()
