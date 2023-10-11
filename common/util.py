import numpy as np

# 输出层处理，分类问题: logits ==> 概率分布
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


# 损失函数
# 交叉熵
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# 计算梯度
def numerical_gradient1(f, x):
    h = 1e-4
    # 创建一个和 x 同样形状的零数组来存储梯度
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x +h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x - h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 还原
        x[idx] = tmp_val

    return grad


# 数值微分计算梯度
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # 使用了 NumPy 的 nditer 来提高效率
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        
        # f(x + h)
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) 

        # f(x - h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        # 还原
        x[idx] = tmp_val
        it.iternext()

    return grad


# 图像 ==> 列（矩阵）
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # (0, 0) 在第1,2 维度N，C不填充；相当于在3,4维度 H, W上填充
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')

    # 存储经过滤波操作后的数据；
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # y_max 和 x_max 是根据当前滤波器的位置计算出的卷积窗口的边界
    # 卷积窗口在移动过程中可以重叠
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            # col[:, :, y, x, :, :] 是将卷积操作的结果存储在 col 数组中的特定位置
            # img[:, :, y:y_max:stride, x:x_max:stride] 表示从输入图像 img 中提取数据，以进行卷积操作
            # 这些提取的数据是卷积窗口在输入图像上滑动的结果。 stride 控制了滑动的步幅。
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # transpose(0, 4, 5, 1, 2, 3): 将原始的维度重新排序为 (N, out_h, out_w, C, filter_h, filter_w)
    # reshape: 重塑为一个二维数组,-1 表示让 NumPy 自动计算第二个维度的大小
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


# 列（矩阵） ==> 图像
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]