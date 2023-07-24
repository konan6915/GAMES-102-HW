import numpy as np
import matplotlib

matplotlib.use("Agg")  # 使用Agg后端，不显示图形窗口
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 存储选取的点
selected_points = []

# 默认高斯插值的 sigma 值
default_sigma = 1.0
default_span = 1
default_lambda = 1.0


def interpolation(x_interp, x_points, y_points):
    n = len(x_points)
    m = n - 1

    # 构建系数矩阵 A 和常数向量 b
    # A = np.array([[x_points[i] ** j for j in range(m + 1)] for i in range(n)])
    A = np.array([[x ** i for i in range(m + 1)] for x in x_points])
    b = y_points

    # 解线性方程组，得到系数向量 coefficients
    coefficients = np.linalg.solve(A, b)

    # 构建插值函数
    y_interp = np.zeros_like(x_interp)
    for i, x in enumerate(x_interp):
        y_interp[i] = sum(coefficients[j] * x ** j for j in range(m + 1))

    return y_interp


def interpolation_min(x_interp, x_points, y_points, m=default_span):
    A = np.array([[x ** i for i in range(m)] for x in x_points])
    A_plus = np.dot(A.T, A)
    b = np.dot(A.T, y_points)

    # 解线性方程组，得到系数向量 coefficients
    coefficients = np.linalg.solve(A_plus, b)

    # 构建插值函数
    y_interp = np.zeros_like(x_interp)
    for i, x in enumerate(x_interp):
        y_interp[i] = sum(coefficients[j] * x ** j for j in range(m))

    return y_interp


def Ridge_Regression(x_interp, x_points, y_points, m=default_span, lamb=default_lambda):
    A = np.array([[x ** i for i in range(m)] for x in x_points])
    lamb_identity = lamb * np.identity(m)
    coefficients = np.linalg.inv(np.dot(A.T, A) + lamb_identity).dot(A.T).dot(y_points)

    # 构建插值函数
    y_interp = np.zeros_like(x_interp)
    for i, x in enumerate(x_interp):
        y_interp[i] = sum(coefficients[j] * x ** j for j in range(m))

    return y_interp


# ------------------------------------------------------------------------
# 拉格朗日基函数
def lagrange_basis(x, i, x_points):
    basis = 1
    for j in range(len(x_points)):
        if j != i:
            basis *= (x - x_points[j]) / (x_points[i] - x_points[j])
    return basis


# 拉格朗日插值多项式
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    interpolated_value = 0
    for i in range(n):
        interpolated_value += y_points[i] * lagrange_basis(x, i, x_points)
    return interpolated_value


# --------------------------------------------------------------------------
# 高斯基函数
def gaussian_basis(x, x_points, sigma):
    return np.exp(-((x - x_points) ** 2) / (2 * sigma ** 2))


# 高斯插值多项式
def gauss_interpolation(x_interp, x_points, y_points, sigma=default_sigma):
    if len(x_points) < 2:
        return np.zeros_like(x_interp)

    num_gauss = len(x_points)
    mu_values = x_points

    A = np.zeros((len(x_points), num_gauss))
    for i, mu in enumerate(mu_values):
        A[i] = gaussian_basis(mu, x_points, sigma)

    # 使用线性方程组求解插值函数的系数
    coefficients = np.linalg.solve(A, y_points)

    # 构建插值函数
    y_interp = np.zeros_like(x_interp)
    for i, x in enumerate(x_interp):
        y_interp[i] = np.dot(gaussian_basis(x, x_points, sigma), coefficients)

    return y_interp


# -----------------------------------------------------------------------------
#
# 绘制原始数据点和插值多项式
def plot_interpolation(x_points, y_points, sigma=default_sigma, span=default_span, lamb=default_lambda):
    plt.clf()  # 清空之前的图形，以便更新图像
    # 设置x轴和y轴的范围
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.scatter(x_points, y_points, label="Data Points")

    if len(x_points) >= 2:  # 至少有两个点时进行插值计算和绘图
        x_interp = np.linspace(min(x_points), max(x_points), 100)
        y_interpolation = interpolation(x_interp, x_points, y_points)
        y_gauss = gauss_interpolation(x_interp, x_points, y_points, sigma)
        # y_lagrange = lagrange_interpolation(x_interp, x_points, y_points)
        y_min = interpolation_min(x_interp, x_points, y_points, span)
        y_ridge = Ridge_Regression(x_interp, x_points, y_points, span, lamb)
        plt.plot(x_interp, y_interpolation, color='r', label="Interpolated Polynomial")
        plt.plot(x_interp, y_gauss, color='g', label="Gauss Interpolated Polynomial (sigma={})".format(sigma))
        plt.plot(x_interp, y_min, color='m', label="LeastSquare (span={})".format(span))
        plt.plot(x_interp, y_ridge, color='b', label="Ridge Regression(lambda={})".format(lamb))
        # plt.plot(x_interp, y_lagrange, color='r', label="Lagrange Interpolated Polynomial")

    # 添加图例和标签
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolation Function")

    plt.grid(True)
    plt.savefig("interpolation.png")  # 保存图形文件


# 处理鼠标点击事件，选取点
def onclick(event):
    if event.button == 1:  # 左键点击
        x, y = event.xdata, event.ydata
        selected_points.append((x, y))
        x_points = [point[0] for point in selected_points]
        y_points = [point[1] for point in selected_points]
        plot_interpolation(x_points, y_points, float(sigma_entry.get()), int(span_entry.get()),
                           float(lamb_entry.get()))  # 重新绘制插值多项式


# 创建GUI窗口
root = Tk()
root.title("Curve Fitting")

# 添加输入框和标签
sigma_label = Label(root, text="Enter Gaussian Sigma:")
sigma_label.pack()

sigma_entry = Entry(root)
sigma_entry.pack()
sigma_entry.insert(0, str(default_sigma))  # 设置默认值

span_label = Label(root, text="Enter Interpolation Span:")
span_label.pack()

span_entry = Entry(root)
span_entry.pack()
span_entry.insert(0, str(default_span))  # 设置默认值

lamb_label = Label(root, text="Enter Ridge Lambda:")
lamb_label.pack()

lamb_entry = Entry(root)
lamb_entry.pack()
lamb_entry.insert(0, str(default_span))  # 设置默认值

# 添加绘制按钮
plot_button = Button(root, text="Plot", command=lambda: plot_interpolation([point[0] for point in selected_points],
                                                                           [point[1] for point in selected_points],
                                                                           float(sigma_entry.get()),
                                                                           int(span_entry.get()),
                                                                           float(lamb_entry.get())))
plot_button.pack()


# 添加清除按钮的回调函数
def clear_points():
    global selected_points
    selected_points = []
    plot_interpolation([], [], float(sigma_entry.get()), int(span_entry.get()), float(lamb_entry.get()))


# 添加清除按钮
clear_button = Button(root, text="Clear Points", command=clear_points)
clear_button.pack()

# 连接事件处理函数
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
canvas.mpl_connect('button_press_event', onclick)

# 初始化一个空的坐标轴范围
plt.xlim(0, 50)
plt.ylim(0, 50)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gauss Interpolation")
plt.grid(True)

plt.tight_layout()  # 调整图形布局


# 在Tkinter主窗口关闭事件中关闭Matplotlib图形窗口
def on_closing():
    plt.close(fig)
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)

# 启动GUI窗口的消息循环
root.mainloop()
