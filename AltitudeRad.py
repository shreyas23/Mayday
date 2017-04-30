""" Take in some array of altitudes at some time in chronological 
order and output an array of TID values as time passes"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

january = [82.4, 85.44, 86.31, 77.77, 71.8, 69.9, 66.28, 62.89, 71.85, 78.71]
february = [69.75, 85.73, 82.54, 78.13, 69.75, 70.87, 64.27, 62.86, 73.66, 80.88]
march = [80.74, 86.74, 82.04, 75.6, 65.6, 69.31, 65.75, 62.33, 74.14, 82.69]
april = [81.39, 87.75, 79.72, 72.09, 71.8, 69.6, 65.89, 63.6, 74.19]
may = [80.75, 87.61, 79.58, 74.05, 72.09, 64.48, 67.94, 65.45, 74.48]
june = [80.59, 87.46, 78.71, 69.31, 69.31, 63.77, 64.83, 64.66, 74.58]
july = [80.88, 86.88, 78.13, 70.87, 64.02, 64.62, 65.94, 66.63, 73.85]
august = [81.89, 86.74, 77.05, 71.26, 64.83, 65.31, 70.19, 66.72, 74.92]
september = [83.99, 87.46,  77.63, 70.97, 67.46, 66.38, 65.99, 66.82, 75.6]
october = [83.34, 77.63, 78.13, 70.29, 67.89, 67.89, 66.97, 68.38, 77.34]
november = [84.43, 86.31, 77.41, 71.95, 68.19, 66.38, 65.12, 68.92, 78.49]
december = [84.71, 87.03, 76.69, 73.9, 69.9, 64.48, 62.11, 68.68, 78.93]

month = february  # user input

plt.ion()
X = [[x] for x in list(range(len(month)))]
vector = month
predict = [[x] for x in list(range(len(month) + 3))]

plt.title("february")
plt.plot(X, vector)
plt.show()

poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(X)
predict_ = poly.fit_transform(predict)
clf = linear_model.LinearRegression()
clf.fit(X_, vector)

print(clf.predict(predict_))
plt.plot(clf.predict(predict_))

def main():
    return clf.predict(predict_)

if __name__ == "__main__":
    main();

"""
plt.ion()
n_observations = len(month)
fig, ax = plt.subplots(1, 1)
xs = np.array(range(len(month)))
print(xs)
ys = np.array(month)
print(ys)
ax.scatter(xs, ys)
#fig.show()
#plt.draw()

# variables which we need to fill in when we are ready to compute the graph.
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

Y_pred = tf.Variable(tf.random_normal([1]), name='bias')
for pow_i in range(1, 5):
    W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
    Y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), W), Y_pred)

# %% Loss function will measure the distance between our observations
# and predictions and average over them.
cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

n_epochs = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        for (x, y) in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        training_cost = sess.run(
            cost, feed_dict={X: xs, Y: ys})
        print({X: xs})
        print({Y: ys})
        print(training_cost)

        if epoch_i % 100 == 0:
            ax.plot(xs, Y_pred.eval(
                feed_dict={X: xs}, session=sess),
                    'k', alpha=epoch_i / n_epochs)
            fig.show()
            plt.draw()

        # Allow the training to quit if we've reached a minimum
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            break
        prev_training_cost = training_cost
ax.set_ylim([-3, 3])
fig.show()
plt.waitforbuttonpress()
"""
# Linear Regression -- results were not accurate enough
"""
input_ = tflearn.input_data(shape=[None])
linear = tflearn.single_unit(input_)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=0.01)
num_of_months = list(range(len(april[1:])))
plt.plot(num_of_months, april[1:])
plt.show()

m = tflearn.DNN(regression)
m.fit(num_of_months, april[1:], n_epoch=1500, show_metric=True, snapshot_epoch=False)

print("\nRegression result:")
print("Y = " + str(m.get_weights(linear.W)) +
      "*X + " + str(m.get_weights(linear.b)))

print("\nTest prediction for x = 3.2, 3.3, 3.4:")
print(m.predict([10, 11, 12]))


# Update: Different implementation!
def get_contents(filename):
    with open(filename) as f:
        content = f.readlines()
        content = [x.strip().split(',') for x in content]
    return content

# Returns an R in usv/hr
def alt_to_rad(time_interval, altitudes):
    time_rising = 25
    time_cruising = 481
    time_landing = 37

    #Altitudes in km
    A = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

    R = [((m * x) + 5) for m,x in zip(A, linear_fit)]
"""



