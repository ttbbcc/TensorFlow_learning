import numpy as np
# 该预测只针对线性函数
# y = wx + b
def compute_error_for_line_given_points(b,w,points):
    totalError = 0
    for i in range(0,len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    # average loss for each point
    return totalError / float(len(points))

# 通过梯度和learningRate不断更新b和w的值，从而得到一个较为符合点的拟合曲线
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0,len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # grad_b = 2(wx+b-y)通过对原函数求偏导以及链式法则计算得出
        b_gradient += (2/N) * ((w_current * x + b_current) - y)
        # grad_w = 2(wx+b-y)*x
        w_gradient += (2/N) * ((w_current * x + b_current) - y) * x
        # 通过b_gradient和w_gradient 的值不断更新b和w的值
        # learnignRate 是学习率，用于控制更新步长
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

# 梯度下降算法的实现，用于优化线性回归模型的参数
# num_iterations定义迭代次数
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b,w = step_gradient(b, w, np.array(points), learning_rate) # 调用上一个编写的函数
    return [b, w]


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_w = 0  # initial slope guess
    num_iterations = 10000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points))
          )


if __name__ == '__main__':
    run()



