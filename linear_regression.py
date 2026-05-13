#%% md
#%%
import numpy as np
import scipy.stats as st
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import keyboard
import random

x_tr = np.array(np.linspace(0,1,10))                    # X coordinates of training data
y_tr = np.array([1.87,-1.34,2.77,2.87,3.43,3.87,3.30,4.13,8.18,7.67])   # Y coordinates of training data


def f(x): return np.exp(2*x)
x = np.linspace(0., 2, 200)
y = f(x)

# Linear regression model
lr = lm.LinearRegression()
# Transform the training data in the required format
X_tr = x_tr[:, np.newaxis]
# Train the model on the training data
lr.fit(X_tr, y_tr)

print("w_1 =",lr.coef_)
print("w_0 =",lr.intercept_)

# Just like fit, predict needs the input data to be a n_sample*n_features matrix
X = x[:, np.newaxis]
y_lr = lr.predict(X)

# Let's plot everything
plt.figure(figsize=(6,3))
plt.plot(x,y,'--k')
plt.plot(x_tr,y_tr,'og', ms=5)
plt.plot(x,y_lr,'b')
plt.legend(['Original line','Training points','Linear regression'])
plt.title("Linear regression")
plt.show()

'''
From this simulation you can see that linear regression consists of a series of problem,such as the limitation of
training data in range, it can hardly fits any equation. How ever,By performing repeated linear approximations, 
recording the intersection points of every two adjacent approximate lines (i.e., the intersection points of the tangent lines to the original curve),
 and then connecting each of these adjacent intersection points, a high-cost yet precise linear approximation can be achieved. '''

print("enter c to continue")
keyboard.wait("c")


def f(x): return np.exp(2*x)
x = np.linspace(0., 6.4, 700)  # you can change the x&y range of the original line here
y = f(x)

x_new = np.linspace(0, 6, 300) # you can change the range and number of training points here
y_new = []

for i in (x_new):
    error = random.uniform(-0.3,0.3)

    y_data = np.exp(2 * i) * (1 + error)  #   This time, we induce random error instead of typing every coordinates

    y_new.append(y_data)

y_new=np.array(y_new)

# store slope of every line between points
slopes = []

plt.figure(figsize=(10, 6))

plt.plot(x,y,'--k')
plt.plot(x_new,y_new,'og', ms=4)

# we set the step as 10 ,so Y predict with every 10 points
for i in range(0, len(x_new) - 10, 10):
    segment_x = x_new[i:i + 10, np.newaxis]
    segment_y = y_new[i:i + 10]

    local_lr = lm.LinearRegression()
    local_lr.fit(segment_x, segment_y)


    center_x = (x_new[i] + x_new[i + 9]) / 2

    line_x_extended = np.linspace(center_x-0.2 , center_x + 1.25, 10)

    # draw the regression line every 10 points
    plt.plot(line_x_extended, local_lr.predict(line_x_extended[:, np.newaxis]), 'b', alpha=0.6, linewidth=1)

    slopes.append(local_lr.coef_[0])

plt.legend(['Original line','Training points','Linear regression'])

plt.title("Linear regression")
plt.show()

average_slope=0
count=0
for i in slopes:
    count+=1
    print(f"number {count} of slope of regression line: {int(i)}")
    average_slope+=int(i)

average_slope=average_slope/len(slopes)

print("average slope",average_slope) # you can see that the avg slope will be very large

'''you can change some of the value to see the different'''


'''From this program, you can see that the linear regression can still apply with enough training points 
However the cost is very large with multiple loops in the program,and the predicted result will be very unstable
 once it is out of the range of training data'''



