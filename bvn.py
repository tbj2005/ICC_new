import numpy as np
import copy
import time
import LP_stuff
from scipy.optimize import linear_sum_assignment
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def solve_target_matrix(matrix, size):
    """
    求解满足条件的 target_matrix。

    参数:
        matrix: 输入的矩阵，对角线元素为0。
        size: 矩阵的行数（列数相同）。

    返回:
        target_matrix: 满足条件的矩阵。
        row_sum: 每行每列的和（优化目标）。
    """
    assert matrix.shape == (size, size), "输入矩阵必须是方阵"
    assert np.all(np.diag(matrix) == 0), "输入矩阵的对角线元素必须全为0"

    # 变量：非对角线元素（按行展开） + s（行和）
    num_vars = size * (size - 1) + 1
    c = np.zeros(num_vars)
    c[-1] = 1  # 最小化 s

    # 约束条件：
    # 1. 每行和 = s（共 size 个约束）
    # 2. 每列和 = s（共 size-1 个约束，最后一列依赖行约束）
    # 3. target_matrix[i][j] >= matrix[i][j]（非对角线）
    A_eq_rows = np.zeros((size, num_vars))  # 行约束
    A_eq_cols = np.zeros((size - 1, num_vars))  # 列约束（去掉最后一列）
    A_ub = np.zeros((size * (size - 1), num_vars))  # 元素下限

    # 填充行约束
    for i in range(size):
        for j in range(size):
            if i != j:
                idx = i * (size - 1) + (j if j < i else j - 1)
                A_eq_rows[i, idx] = 1
        A_eq_rows[i, -1] = -1  # -s

    # 填充列约束（去掉最后一列）
    for j in range(size - 1):
        for i in range(size):
            if i != j:
                idx = i * (size - 1) + (j if j < i else j - 1)
                A_eq_cols[j, idx] = 1
        A_eq_cols[j, -1] = -1  # -s

    # 合并等式约束
    A_eq = np.vstack([A_eq_rows, A_eq_cols])
    b_eq = np.zeros(size + (size - 1))

    # 填充元素下限约束
    k = 0
    for i in range(size):
        for j in range(size):
            if i != j:
                idx = i * (size - 1) + (j if j < i else j - 1)
                A_ub[k, idx] = -1
                k += 1
    b_ub = -matrix[~np.eye(size, dtype=bool)].flatten()

    # 求解
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=(0, None), method='highs'
    )

    if not res.success:
        raise ValueError("线性规划求解失败: " + res.message)

    # 构造结果矩阵
    target_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j:
                idx = i * (size - 1) + (j if j < i else j - 1)
                target_matrix[i, j] = res.x[idx]
    s = res.x[-1]

    return target_matrix, s


def solve_target_matrix_optimized(matrix, size):
    """
    优化后的求解函数，保持相同功能但提高速度。
    """
    assert matrix.shape == (size, size), "输入矩阵必须是方阵"
    assert np.all(np.diag(matrix) == 0), "输入矩阵的对角线元素必须全为0"

    # 变量：非对角线元素（按行展开） + s（行和）
    num_vars = size * (size - 1) + 1
    c = np.zeros(num_vars)
    c[-1] = 1  # 最小化 s

    # 预计算非对角线元素的索引
    row_indices, col_indices = np.where(~np.eye(size, dtype=bool))
    var_indices = np.arange(size * (size - 1)).reshape(size, size - 1)

    # 构造行约束矩阵 - 向量化方法
    A_eq_rows = np.zeros((size, num_vars))
    for i in range(size):
        cols = [j for j in range(size) if j != i]
        var_idx = [var_indices[i, j - (1 if j > i else 0)] for j in cols]
        A_eq_rows[i, var_idx] = 1
    A_eq_rows[:, -1] = -1  # -s 项

    # 构造列约束矩阵 - 向量化方法 (去掉最后一列)
    A_eq_cols = np.zeros((size - 1, num_vars))
    for j in range(size - 1):
        rows = [i for i in range(size) if i != j]
        var_idx = [var_indices[i, j - (1 if j > i else 0)] for i in rows]
        A_eq_cols[j, var_idx] = 1
    A_eq_cols[:, -1] = -1  # -s 项

    # 合并等式约束
    A_eq = np.vstack([A_eq_rows, A_eq_cols])
    b_eq = np.zeros(A_eq.shape[0])

    # 元素下限约束 - 向量化构造
    A_ub = np.zeros((size * (size - 1), num_vars))
    diag_mask = ~np.eye(size, dtype=bool)
    A_ub[np.arange(size * (size - 1)), np.arange(size * (size - 1))] = -1
    b_ub = -matrix[diag_mask].flatten()

    # 求解 - 使用更高效的求解器参数
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=(0, None), method='highs',
        options={"disp": False, "time_limit": 60}
    )

    if not res.success:
        raise ValueError("线性规划求解失败: " + res.message)

    # 构造结果矩阵
    target_matrix = np.zeros((size, size))
    for i, j in zip(row_indices, col_indices):
        idx = var_indices[i, j - (1 if j > i else 0)]
        target_matrix[i, j] = res.x[idx]
    s = res.x[-1]

    return target_matrix, s


def stuffing_min(matrix, R_S):
    copied_matrix = np.copy(matrix)
    while True:
        row_sum = copied_matrix.sum(axis=1)
        column_sum = copied_matrix.sum(axis=0)
        if np.array_equal(row_sum, column_sum):
            if np.all(row_sum == row_sum[0]) and np.all(column_sum == column_sum[0]):
                break
        max_value = max(np.maximum(row_sum, column_sum))
        flag_column = False  #寻找列为假
        row_differ = np.array([max_value] * R_S) - row_sum  #获取行的差
        column_differ = np.array([max_value] * R_S) - column_sum  #获取列的差

        s_R_d_index = np.argsort(row_differ)  #行差排序后的索引
        sort_row_differ = row_differ[s_R_d_index]  #行差排序后的结果

        s_C_d_index = np.argsort(column_differ)  #列差排序后的索引
        sort_column_differ = column_differ[s_C_d_index]  #列差排序后的结果

        # 合并排序后的行差和列差为一个新的数组
        combined_array = np.concatenate((sort_row_differ.reshape(1, -1), sort_column_differ.reshape(1, -1)), axis=1)

        # 找到新数组非零元素的索引
        nonzero_indices = np.nonzero(combined_array)

        # 获取新数组非零元素
        nonzero_elements = combined_array[nonzero_indices]

        # 找到新数组非零最小值的索引和非零元素的最小值
        min_nonzero_index = np.argmin(nonzero_elements)
        min_nonzero_d = nonzero_elements[min_nonzero_index]  # 这个值就是用来填充的值

        # 返回这个数在原矩阵中的索引
        min_d_index = nonzero_indices[1][min_nonzero_index]
        # max_d_index = nonzero_indices[1][max_nonzero_index]

        # if max_d_index < R_S or min_d_index < R_S:
        if min_d_index < R_S:
            # max_p = s_R_d_index[max_d_index]
            min_p = s_R_d_index[min_d_index]  # 最小差距是行
            flag_column = True  # 此时要寻找最大差距的列
        else:
            min_p = s_C_d_index[min_d_index % R_S]  # 最小车距是列，此时要寻找最大差距的行
            # max_p = s_C_d_index[max_d_index % R_S]

        if flag_column:  # 如果要寻找最大差距的列
            max_to_stuff_index = np.argmax(column_differ)  # 获取差最大值的索引
            if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:
                copied_matrix[min_p][max_to_stuff_index] += min_nonzero_d
            else:
                if min_p != max_to_stuff_index:
                    copied_matrix[min_p][max_to_stuff_index] += min_nonzero_d
                else:
                    column_differ[max_to_stuff_index] = np.iinfo(np.int32).min
                    # 找到第二大值的索引
                    second_largest_index = np.argmax(column_differ)
                    copied_matrix[min_p][second_largest_index] += min_nonzero_d
            # 根据索引信息找到在原数组中的索引
        else:
            max_to_stuff_index = np.argmax(row_differ)  # 获取差最大值的索引
            if np.count_nonzero(row_differ) == 1 or np.count_nonzero(column_differ) == 1:
                copied_matrix[max_to_stuff_index][min_p] += min_nonzero_d
            else:
                if min_p != max_to_stuff_index:
                    copied_matrix[max_to_stuff_index][min_p] += min_nonzero_d
                else:
                    row_differ[max_to_stuff_index] = np.iinfo(np.int32).min
                    # 找到第二大值的索引
                    second_largest_index = np.argmax(row_differ)
                    copied_matrix[second_largest_index][min_p] += min_nonzero_d
    return copied_matrix


def max_component(matrix_stuff, matrix_data, size):
    """
    找到该矩阵的可分解最大权置换矩阵
    :param matrix_stuff:
    :param matrix_data:
    :param size:
    :return:
    """
    nonzero_num = np.count_nonzero(matrix_stuff)
    nonzero_num_data = np.count_nonzero(np.where(copy.deepcopy(matrix_data) >= 0.000000001, matrix_data, 0))
    sort_index = np.argsort(-1 * matrix_data, axis=None)[: nonzero_num_data]
    sort_row_col = [(int(sort_index[i] / size), int(sort_index[i] % size)) for i in range(len(sort_index))]
    reserve_matrix = np.where(matrix_data > 0, 0, copy.deepcopy(matrix_stuff))
    sort_index_stuff = np.argsort(-1 * reserve_matrix, axis=None)[: nonzero_num - nonzero_num_data]
    sort_row_col_stuff = [(int(sort_index_stuff[i] / size), int(sort_index_stuff[i] % size)) for i in range(len(sort_index_stuff))]
    bool_matrix = np.zeros([size, size])
    ava_row = [i for i in range(size)]
    ava_col = [i for i in range(size)]
    row_ind = []
    col_ind = []
    flag = 0
    value = 0
    for i in range(nonzero_num):
        if i >= nonzero_num_data:
            (sort_row, sort_col) = sort_row_col_stuff[i - nonzero_num_data]
        else:
            (sort_row, sort_col) = sort_row_col[i]
        if sort_row in ava_row:
            ava_row = [i for i in ava_row if i != sort_row]
        if sort_col in ava_col:
            ava_col = [i for i in ava_col if i != sort_col]
        bool_matrix[sort_row][sort_col] = 1  # 将元素布尔化，方便使用最大流
        if len(ava_row) + len(ava_col) > 0:
            # 如果行列没填满，一定是不会有非零最大匹配的
            continue
        else:
            row_ind, col_ind = linear_sum_assignment(bool_matrix, maximize=True)
            match_ele = np.array([bool_matrix[row_ind[j]][col_ind[j]] for j in range(len(row_ind))])
            # 对布尔矩阵求最大匹配，绕开元素大小，只看是否有 0
            if np.min(match_ele) == 0:
                # 有 0 说明当前还没有非零最大匹配
                continue
            else:
                value = np.min(np.array([matrix_stuff[row_ind[c]][col_ind[c]] for c in range(len(row_ind))]))
                flag = 1
                break
    if flag == 0:
        return np.zeros([size, size])
    else:
        max_component_matrix = np.zeros([size, size])
        for i in range(len(row_ind)):
            max_component_matrix[row_ind[i]][col_ind[i]] = value
        return max_component_matrix


def diagonal_zero_stuff(raw_matrix, size):
    """
    将对角元素为 0 的矩阵进行填充，使其变为双随机矩阵。
    :param raw_matrix:
    :param size:
    :return:
    """
    matrix = copy.deepcopy(raw_matrix)
    max_ele = np.max(matrix)
    full_matrix = max_ele * np.ones([size, size]) - max_ele * np.eye(size)
    add_matrix = full_matrix - matrix
    while 1:
        add_component = max_component(add_matrix, size)
        if np.max(add_component) == 0:
            break
        add_matrix -= add_component
    print(np.sum(matrix + add_matrix) / size)
    return matrix + add_matrix


def is_strongly_connected_scipy(adj_matrix):
    n = adj_matrix.shape[0]
    if n == 0:
        return True

    # 转换为稀疏矩阵（仅关心是否有连接）
    graph = csr_matrix((adj_matrix > 0).astype(int))
    n_components, _ = connected_components(graph, directed=True, connection='strong')
    return n_components == 1  # 强连通分量是否为1


def is_routing_supported(data_matrix, connection_matrix):
    """
    检查连接矩阵是否支持数据矩阵在两条路径内的所有转发需求

    参数:
    data_matrix: numpy.ndarray - 非负数据量矩阵，对角线不为0
    connection_matrix: numpy.ndarray - 0-1连接矩阵

    返回:
    bool - 如果支持所有转发需求返回True，否则返回False
    """
    # 验证输入矩阵的形状是否相同
    assert data_matrix.shape == connection_matrix.shape, "输入矩阵形状必须相同"
    n = data_matrix.shape[0]

    # 检查对角线是否都不为0
    assert np.all(np.diag(data_matrix) == 0), "数据矩阵对角线元素必须不为0"

    # 对于每对节点(i,j)，检查是否能直接连接或通过一个中间节点k连接
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # 对角线不需要转发

            required_capacity = data_matrix[i, j]
            if required_capacity == 0:
                continue  # 没有数据传输需求

            # 检查直接连接
            direct_capacity = connection_matrix[i, j]
            if direct_capacity >= 1:
                continue  # 直接连接满足需求

            # 检查是否存在中间节点k使得i->k和k->j都有连接
            found = False
            for k in range(n):
                if k == i or k == j:
                    continue
                if connection_matrix[i, k] >= 1 and connection_matrix[k, j] >= 1:
                    found = True
                    break

            if not found:
                # print(f"无法满足从节点 {i} 到节点 {j} 的数据转发需求")
                return False

    return True


def matrix_decompose(data_matrix, matrix, size, threshold, num):
    """
    矩阵分解策略
    :param data_matrix:
    :param matrix:
    :param size:
    :param threshold: 分解数
    :param num: 分解矩阵的最大个数
    :return:
    """
    decompose_matrix = []
    use_matrix = copy.deepcopy(matrix)
    use_matrix = use_matrix.astype(np.float64)
    finish_data = np.zeros([size, size])
    data_matrix_copy = copy.deepcopy(data_matrix)
    use_data_rate = 0
    count = 0
    while 1:
        # 第一轮分解，无需填充
        # if count == num and num > 0:
        #     break
        if len(decompose_matrix) == num:
            break
        max_matrix = max_component(use_matrix, data_matrix_copy - finish_data, size)
        max_matrix = max_matrix.astype(np.float64)
        bool_finish = np.where(finish_data > 0, 1, 0)
        if np.max(max_matrix) == 0 or is_routing_supported(data_matrix, bool_finish):
            if use_data_rate >= threshold:
                break
        decompose_matrix.append(max_matrix)
        use_matrix -= max_matrix
        finish_data += max_matrix
        use_data_rate = np.sum(np.minimum(finish_data, data_matrix)) / np.sum(data_matrix)
        count += 1
    return decompose_matrix, finish_data

# matrix_size = 4
# matrix_test = np.random.randint(0, 100, size=(matrix_size, matrix_size))  # 生成随机正整数矩阵，取值范围为1到10
# np.fill_diagonal(matrix_test, 0)  # 将对角线元素设为0
# # print("原矩阵：\n", matrix_test)
# start1 = time.time()
# stuff_matrix = diagonal_zero_stuff(matrix_test, matrix_size)
# mid1 = time.time()
# print(mid1 - start1)
# # print("填充矩阵：\n", stuff_matrix)
# # decompose_ele = matrix_decompose(stuff_matrix, matrix_size)
# # end = time.time()
# # # print(mid - start, end - mid)
# # print("分解矩阵：\n", decompose_ele)
# start = time.time()
# lp_matrix = LP_stuff.lp_stuff(matrix_test, matrix_size)
# end = time.time()
# print(np.sum(lp_matrix) / matrix_size)
# print(end - start)
#
# start = time.time()
# lp_matrix_2 = stuffing_min(matrix_test, matrix_size)
# print(np.sum(lp_matrix_2) / matrix_size)
# end = time.time()
# print(end - start)
#
# start = time.time()
# lp_matrix_3, _ = solve_target_matrix(matrix_test, matrix_size)
# # print(lp_matrix_3)
# print(np.sum(lp_matrix_3) / matrix_size)
# end = time.time()
# print(end - start)
