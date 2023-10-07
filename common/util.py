import numpy as np

# 图像展开为矩阵
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