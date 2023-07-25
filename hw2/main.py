import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # 使用Agg后端，不显示图形窗口
from tkinter import Tk, Label, Entry, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

default_epoch = 5000
default_lr = 0.01
default_size = 20

# 存储选取的点
selected_points = []

# 创建 RBF 网络实例
rbf_model = None


class RBFKernelStd(nn.Module):
    def __init__(self):
        super(RBFKernelStd, self).__init__()
        self.coefficient = 1 / np.sqrt(2 * np.pi)

    def forward(self, x):
        return torch.exp(-0.5 * x * x) * self.coefficient


class RBF(nn.Module):
    def __init__(self, n_params=1000):
        super(RBF, self).__init__()
        # n_params 每个 RBF 核函数的数量
        self.n_params = n_params
        # 高斯核
        self.kernel = RBFKernelStd()
        # 扩展 RBF 输出的张量
        self.one = torch.Tensor([1.])
        # torch.ones(self.n_params)：创建一个大小为 self.n_params 的张量，其中每个元素都初始化为 1。这些元素将用作参数 a 的初始值
        # requires_grad=True：将参数 a 设置为需要计算梯度
        self.a = nn.Parameter(torch.ones(self.n_params), requires_grad=True)
        self.b = nn.Parameter(torch.ones(self.n_params), requires_grad=True)
        self.linear = nn.Linear(n_params, 1, bias=True)
        # self.w = nn.Parameter(torch.ones(self.n_params + 1), requires_grad=True)
        self.init()

    def init(self):
        self.a.data.normal_(0, 0.2)
        self.b.data.normal_(0, 0.2)
        self.linear.weight.data.normal_(0, 0.2)

    def forward(self, x):
        g = self.kernel(self.a * x + self.b)
        y = self.linear(g)
        return y


def train_rbf_model(model, x, y, lr=default_lr, epochs=default_epoch):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    print("Training finished!")


# -----------------------------------------------------------------------------
def plot_interpolation(x_points, y_points):
    plt.clf()  # 清空之前的图形，以便更新图像
    # 设置x轴和y轴的范围
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.scatter(x_points, y_points, label="Data Points")

    if len(x_points) >= 2:  # 至少有两个点时进行插值计算和绘图
        x_interp = np.linspace(min(x_points), max(x_points), 100)
        # 转换为PyTorch张量
        x_train = torch.tensor(x_interp, dtype=torch.float32).unsqueeze(1)
        # y_train = torch.tensor(x_, dtype=torch.float32).unsqueeze(1)

        if rbf_model is not None:
            # 使用训练好的模型对训练数据进行预测
            y_pred = rbf_model(x_train)
            plt.plot(x_train.numpy(), y_pred.detach().numpy(), color='r', label="RBF Curve Fitting")
            # plt.plot(x_interp, y_lagrange, color='r', label="Lagrange Interpolated Polynomial")

    # 添加图例和标签
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Fitting Function")

    plt.grid(True)
    plt.savefig("fitting.png")  # 保存图形文件


# 处理鼠标点击事件，选取点
def onclick(event):
    if event.button == 1:  # 左键点击
        x, y = event.xdata, event.ydata
        selected_points.append((x, y))
        x_points = [point[0] for point in selected_points]
        y_points = [point[1] for point in selected_points]
        plot_interpolation(x_points, y_points)  # 重新绘制插值多项式


# 创建GUI窗口
root = Tk()
root.title("Curve Fitting")

# 添加输入框和标签
epoch_label = Label(root, text="Enter epoch:")
epoch_label.pack()

epoch_entry = Entry(root)
epoch_entry.pack()
epoch_entry.insert(0, str(default_epoch))  # 设置默认值

lr_label = Label(root, text="Enter lr:")
lr_label.pack()

lr_entry = Entry(root)
lr_entry.pack()
lr_entry.insert(0, str(default_lr))  # 设置默认值

size_label = Label(root, text="Enter size:")
size_label.pack()

size_entry = Entry(root)
size_entry.pack()
size_entry.insert(0, str(default_size))  # 设置默认值

# 添加绘制按钮
plot_button = Button(root, text="Plot", command=lambda: train_and_plot(int(epoch_entry.get()), float(lr_entry.get()), int(size_entry.get())))

plot_button.pack()


# 添加清除按钮的回调函数
def clear_points():
    global selected_points, rbf_model
    selected_points = []
    rbf_model = None  # 重置模型为None
    plot_interpolation([], [])


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
plt.title("RBF Fitting")
plt.grid(True)

plt.tight_layout()  # 调整图形布局


# 在Tkinter主窗口关闭事件中关闭Matplotlib图形窗口
def on_closing():
    global selected_points, rbf_model
    plt.close(fig)
    rbf_model = None
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)


def train_and_plot(epochs, lr, size):
    global rbf_model
    # 第一次点击"Plot"按钮时创建 RBF 网络实例
    if rbf_model is None:
        rbf_model = RBF(n_params=size)

    # 转换为PyTorch张量
    x_points = [point[0] for point in selected_points]
    y_points = [point[1] for point in selected_points]
    x_train = torch.tensor(x_points, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_points, dtype=torch.float32).unsqueeze(1)

    # 训练模型
    train_rbf_model(rbf_model, x_train, y_train, lr=lr, epochs=epochs)

    # 绘制拟合曲线
    plot_interpolation(x_points, y_points)
# 启动GUI窗口的消息循环
root.mainloop()
