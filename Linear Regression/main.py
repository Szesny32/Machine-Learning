import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Reshape data -> sklearn need vertical format
exam_time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(-1,1)
exam_scores = np.array([56, 83, 47, 93, 47, 82, 45, 78, 55, 67, 57, 4, 12]).reshape(-1,1)

model = LinearRegression()
model.fit(exam_time_studied, exam_scores)


# A scatter plot of y vs. x with varying marker size and/or color.
plt.scatter(exam_time_studied, exam_scores)
plt.ylim(0,100)
plt.title("Base data")
plt.show()


# plot with linear regression line
# linspace parameters:
# (1) start - The starting value of the sequence.
# (2) stop - The end value of the sequence, unless endpoint is set to False.
# (3) start - Number of samples to generate. Default is 50.

plt.scatter(exam_time_studied, exam_scores)
plt.plot(np.linspace(0,70,100).reshape(-1,1), model.predict(np.linspace(0,70,100).reshape(-1,1)), 'r')
plt.ylim(0,100)
plt.title("Plot with linear regression line")
plt.show()

# score if you study for 56 hours
print("Score if you study for 56 hours: ", model.predict(np.array([56]).reshape(-1,1)))


# we want to train and score data using skleran.model_selection
# in example: 70% per training and 30% per test

exam_time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(-1,1)
exexam_scores = np.array([56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57, 4, 89]).reshape(-1,1)
exam_time_studied_train, exam_time_studied_test, exam_score_train, exam_score_test = train_test_split(exam_time_studied, exexam_scores, test_size=0.4)

best_model = LinearRegression()
best_model.fit(exam_time_studied_train, exam_score_train)
best_score = best_model.score(exam_time_studied_test, exam_score_test) * 100

for i in range(1,10001):
    exam_time_studied_train, exam_time_studied_test, exam_score_train, exam_score_test = train_test_split(exam_time_studied, exexam_scores, test_size=0.4)
    model = LinearRegression()
    model.fit(exam_time_studied_train, exam_score_train)
    score = model.score(exam_time_studied_test, exam_score_test) * 100
    if score > best_score :
        best_score = score
        best_model = model
    print(i, score, "%")

# Listing the coefficients of f(x) = ax + b for the best model
print("Best model: ", best_score, "%")

print("f(x) = ", best_model.coef_[0][0], "x + ", best_model.intercept_[0])
plt.scatter(exam_time_studied_train, exam_score_train)
plt.plot(np.linspace(0,70,100).reshape(-1,1), best_model.predict(np.linspace(0,70,100).reshape(-1,1)), 'r')
plt.title("Best of 10000 with with training and test data ")
plt.show()
