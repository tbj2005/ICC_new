import copy
import time
import ILP_new
import random
import cassini_schedule
import sjf_schedule
import FIFO_schedule
import las_schedule
import hrrn_schedule
import generate_job

import numpy as np
from openpyxl import Workbook
import bvn
import Schedule_part
from scipy.optimize import linear_sum_assignment
import heapq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# 确定业务成环方案
def extract_sub_matrix(matrix, indices):
    """
    从方阵中提取指定行和列组成的新方阵。

    Parameters:
        matrix (np.ndarray): 输入的方阵。
        indices (list): 需要提取的行和列的索引列表。

    Returns:
        np.ndarray: 提取后的新方阵。
    """
    # 使用 np.ix_ 生成行和列的索引网格
    row_col_indices = np.ix_(indices, indices)
    # 提取子矩阵
    sub_matrix = matrix[row_col_indices]
    return sub_matrix


def expand_sub_matrix(sub_matrix, n, indices):
    """
    将子矩阵的值填充回一个全零的 n x n 矩阵的指定位置。

    Parameters:
        sub_matrix (np.ndarray): 输入的子矩阵。
        n (int): 原始方阵的维度。
        indices (list): 子矩阵对应的行和列的索引列表。

    Returns:
        np.ndarray: 填充后的 n x n 矩阵。
    """
    # 创建全零矩阵
    reconstructed = np.zeros((n, n), dtype=sub_matrix.dtype)

    # 使用高级索引填充子矩阵的值
    # indices 的行和列网格
    row_indices = np.array(indices).reshape(-1, 1)  # 列向量
    col_indices = np.array(indices)  # 行向量

    # 将 sub_matrix 的值填充到 reconstructed 的指定位置
    reconstructed[row_indices, col_indices] = sub_matrix

    return reconstructed


def job_ring(job_set_i, fj, ufj, local_solution, single_traffic, sum_traffic, pod, pod_set_i, num_bvn):
    """
    确定业务成环方案
    :param num_bvn:
    :param pod:
    :param pod_set_i:
    :param job_set_i:
    :param sum_traffic: 各业务总流量
    :param fj: 固定拓扑业务索引
    :param ufj: 环拓扑业务索引
    :param local_solution: 业务放置方案
    :param single_traffic: 业务单连接流量
    :return: 返回所有业务的流量矩阵
    """
    link_job_index = [[[] for _ in range(pod)] for _ in range(pod)]
    data_matrix_all = np.zeros(([pod, pod]))
    data_matrix_job = np.array([np.zeros([pod, pod]) for _ in range(len(local_solution))])
    time5 = time.time()
    for i in fj:
        worker_fj = local_solution[i]
        for j in range(len(worker_fj) - 1):
            data_matrix_job[i][worker_fj[j]][worker_fj[j + 1]] += single_traffic[i]
            data_matrix_job[i][worker_fj[j + 1]][worker_fj[j]] += single_traffic[i]
            link_job_index[worker_fj[j]][worker_fj[j + 1]].append(i)
            link_job_index[worker_fj[j + 1]][worker_fj[j]].append(i)
        data_matrix_all += data_matrix_job[i]
    sum_traffic_ufj = np.array([sum_traffic[i] for i in job_set_i if i in ufj])
    time6 = time.time()
    # print("fj", time6 - time5)
    for i in range(len(ufj)):
        job_index = ufj[np.argmax(sum_traffic_ufj)]
        data_matrix_block = extract_sub_matrix(data_matrix_all, pod_set_i)
        time8 = time.time()
        data_matrix_stuff, _ = bvn.solve_target_matrix(data_matrix_block, len(pod_set_i))
        time9 = time.time()
        bvn_compose, sum_bvn = bvn.matrix_decompose(data_matrix_block, data_matrix_stuff, len(pod_set_i), 0.8, num_bvn)
        time10 = time.time()
        # print(i, time9 - time8, time10 - time9)
        sum_bvn_reconstruct = expand_sub_matrix(sum_bvn, pod, pod_set_i)
        worker = local_solution[job_index]
        worker_matrix = np.zeros([len(worker), len(worker)])
        for u in range(len(worker)):
            for v in range(len(worker)):
                if u != v:
                    worker_matrix[u][v] = sum_bvn_reconstruct[worker[u]][worker[v]]
                if u == v:
                    worker_matrix[u][v] = - np.inf
        if np.shape(worker_matrix)[0] == 1:
            data_matrix_job[job_index] = np.zeros([pod, pod])
        else:
            row_ind, col_ind = linear_sum_assignment(worker_matrix, maximize=True)
            data_matrix_single = np.zeros([pod, pod])
            for u in range(len(row_ind)):
                data_matrix_single[worker[row_ind[u]]][worker[col_ind[u]]] = single_traffic[job_index]
                # link_job_index[worker[row_ind[u]]][worker[col_ind[u]]].append(job_index)
            data_matrix_single = sub_ring_edit(data_matrix_single, pod)
            data_matrix_single_zip = extract_sub_matrix(data_matrix_single, pod_set_i)
            data_matrix_single_zip = sub_ring_edit(data_matrix_single_zip, len(pod_set_i))
            data_matrix_single = expand_sub_matrix(data_matrix_single_zip, pod, pod_set_i)
            row_ind_edit, col_ind_edit = np.where(data_matrix_single > 0)
            for u in range(len(row_ind_edit)):
                link_job_index[row_ind_edit[u]][col_ind_edit[u]].append(job_index)
            data_matrix_job[job_index] = data_matrix_single
        sum_traffic_ufj[np.argmax(sum_traffic_ufj)] = -1
        data_matrix_all += data_matrix_job[job_index]

    time7 = time.time()
    # print("ufj", time7 - time6)
    return data_matrix_job, link_job_index


# TPE 方案
def is_strongly_connected_scipy(adj_matrix):
    """
    使用SciPy检查强连通性（推荐）
    """
    n = adj_matrix.shape[0]
    if n == 0:
        return True

    # 转换为稀疏矩阵（仅需连接是否存在）
    graph = csr_matrix((adj_matrix > 0).astype(int))
    n_components, _ = connected_components(graph, directed=True, connection='strong')
    return n_components == 1


def sort_indices_desc(matrix, num):
    # 获取矩阵的尺寸
    n = matrix.shape[0]

    # 将矩阵展平并获取排序后的索引
    flattened = matrix.flatten()
    sorted_indices = np.argsort(flattened)[::-1]  # 降序排序索引

    # 输出前N个索引
    top_n_indices = sorted_indices[:num]

    # 将前N个一维索引转换为二维索引
    indices = np.unravel_index(top_n_indices, (n, n))

    return indices[0], indices[1]


def port_allocate(bvn_compose, port, pod):
    if len(bvn_compose) > port:
        return np.zeros([pod, pod])
    else:
        value_match = np.array([np.sum(bvn_compose[i]) / pod for i in range(len(bvn_compose))])
        match_degree = np.ones(len(bvn_compose))
        count = port - len(bvn_compose)
        while 1:
            if count == 0:
                link_matrix = np.zeros([pod, pod])
                for i in range(len(match_degree)):
                    bool_matrix = np.where(bvn_compose[i] > 0, 1, 0)
                    link_matrix += match_degree[i] * bool_matrix
                return link_matrix
            max_value, max_index = np.max(value_match), np.argmax(value_match)
            match_degree[max_index] += 1
            value_match[max_index] = np.sum(bvn_compose[max_index]) / (pod * match_degree[max_index])
            count -= 1


def max_min_weight_path_dijkstra_no_cycle(adj_matrix, start, end):
    n = adj_matrix.shape[0]
    max_min_heap = [(-float('inf'), start, [start])]  # (-当前路径最小权重, 当前节点, 路径)

    while max_min_heap:
        current_min_neg, u, path = heapq.heappop(max_min_heap)
        current_min = -current_min_neg

        if u == end:
            return current_min, path

        for v in range(n):
            if adj_matrix[u][v] > 0 and v not in path:  # 禁止重复访问节点
                new_min = min(current_min, adj_matrix[u][v])
                heapq.heappush(max_min_heap, (-new_min, v, path + [v]))

    return None, []  # 不可达


import heapq


def max_min_weight_path_dijkstra_k_hops(adj_matrix, start, end, max_hops):
    """
    找到从start到end的路径，使得路径中的最小边权重最大，且路径跳数不超过max_hops

    参数:
        adj_matrix: 邻接矩阵，adj_matrix[u][v]表示u到v的边权重，0表示没有连接
        start: 起始节点索引
        end: 目标节点索引
        max_hops: 允许的最大跳数（边数）

    返回:
        (max_min_weight, path) 元组，找不到路径时返回(None, [])
    """
    n = adj_matrix.shape[0]
    # 优先队列元素: (-当前路径最小权重, 当前节点, 路径, 已用跳数)
    max_min_heap = [(-float('inf'), start, [start], 0)]

    while max_min_heap:
        current_min_neg, u, path, hops = heapq.heappop(max_min_heap)
        current_min = -current_min_neg

        if u == end:
            return current_min, path

        # 如果已经达到最大跳数，不再继续扩展
        if hops >= max_hops:
            continue

        for v in range(n):
            if adj_matrix[u][v] > 0 and v not in path:  # 禁止重复访问节点
                new_min = min(current_min, adj_matrix[u][v])
                new_hops = hops + 1
                heapq.heappush(max_min_heap,
                               (-new_min, v, path + [v], new_hops))

    return None, []  # 不可达


def binary_search(t_low, t_high, job_matrix, sum_matrix, link_matrix, band_per_port, traffic_size, t_threshold,
                  job_link_match, pod_set_b, job_set_b):
    """
    二分查找理想时间
    :param job_set_b:
    :param pod_set_b:
    :param job_link_match:
    :param t_threshold:
    :param traffic_size:
    :param sum_matrix:
    :param band_per_port:
    :param t_low:
    :param t_high:
    :param job_matrix:
    :param link_matrix:
    :return:
    """
    job_matrix_out = copy.deepcopy(job_matrix)
    sum_matrix_out = copy.deepcopy(sum_matrix)
    count = 0
    while 1:
        if t_high - t_low <= t_threshold:
            # print(t_high)
            # print(job_matrix_out)
            return job_matrix_out, sum_matrix_out, count
        sum_matrix_edit = copy.deepcopy(sum_matrix)
        job_matrix_edit = copy.deepcopy(job_matrix)
        t_mid = (t_low + t_high) / 2
        ideal_matrix = t_mid * band_per_port * link_matrix
        delta_matrix = ideal_matrix - sum_matrix_edit
        stuff_decompose = np.where(delta_matrix > 0, delta_matrix, 0)
        reserve_decompose = np.where(delta_matrix < 0, - delta_matrix, 0)
        reverse_row, reverse_col = sort_indices_desc(reserve_decompose, np.count_nonzero(reserve_decompose))
        job_index_sort = np.argsort(- np.array(traffic_size))
        flag_i = 0
        for i in range(np.count_nonzero(reserve_decompose > 0)):
            row, col = reverse_row[i], reverse_col[i]
            job_set_link = job_link_match[row][col]
            for j in range(len(job_index_sort)):
                if reserve_decompose[row][col] <= 0:
                    break
                job_index = job_index_sort[j]
                value = job_matrix_edit[job_index][row][col]
                if job_index in job_set_link:
                    value_path, path = max_min_weight_path_dijkstra_k_hops(stuff_decompose, row, col, 2)
                    if len(path) == 0:
                        break
                    flag = 0
                    if value_path < value:
                        flag = 1
                    if flag == 1:
                        continue
                    else:
                        job_matrix_edit[job_index][row][col] = 0
                        for k in range(len(path) - 1):
                            stuff_decompose[path[k]][path[k + 1]] -= value
                            job_matrix_edit[job_index][path[k]][path[k + 1]] += value
                            sum_matrix_edit[path[k]][path[k + 1]] += value
                        reserve_decompose[row][col] -= value
                        sum_matrix_edit[row][col] -= value
                        continue
            if reserve_decompose[row][col] > 0:
                t_low = t_mid
                flag_i = 1
                break
        if flag_i == 0:
            count += 1
            t_high = t_mid
            job_matrix_out = copy.deepcopy(job_matrix_edit)
            sum_matrix_out = copy.deepcopy(sum_matrix_edit)


def sub_ring_edit(matrix, pod):
    row, _ = np.where(matrix > 0)
    reverse_pod = [i for i in range(pod) if i in row]
    ring = []
    while 1:
        sub_ring = []
        if len(reverse_pod) == 0:
            break
        sub_ring.append(reverse_pod[0])
        while 1:
            next_node = np.argmax(matrix[sub_ring[-1]])
            if next_node in sub_ring:
                break
            else:
                sub_ring.append(next_node)
        reverse_pod = [i for i in reverse_pod if i not in sub_ring]
        ring.append(sub_ring)
    edit_matrix = np.zeros([pod, pod])
    if len(ring) == 1:
        edit_matrix = copy.deepcopy(matrix)
        return edit_matrix
    edit_ring = []
    for i in range(len(ring)):
        edit_ring += ring[i]
    for i in range(len(edit_ring) - 1):
        edit_matrix[edit_ring[i]][edit_ring[i + 1]] = np.max(matrix)
    edit_matrix[edit_ring[-1]][edit_ring[0]] = np.max(matrix)
    return edit_matrix


def tpe(job_set_tpe, job_matrix, job_link_index, port, pod, num_bvn, band_per_port, single_traffic, pod_set_tpe,
        t_threshold):
    """
    tpe 策略
    :param t_threshold:
    :param job_set_tpe:
    :param pod_set_tpe:
    :param single_traffic:
    :param band_per_port:
    :param num_bvn:
    :param job_link_index: 各连接上存在业务的索引
    :param pod: pod 数目
    :param job_matrix: 总流量矩阵
    :param port: port 数目
    :return:
    """
    sum_data_matrix = np.zeros([pod, pod])
    for i in range(len(job_matrix)):
        sum_data_matrix += job_matrix[i]
    job_matrix_edit = copy.deepcopy(job_matrix)
    sum_data_zip = extract_sub_matrix(sum_data_matrix, pod_set_tpe)
    data_matrix_stuff, _ = bvn.solve_target_matrix(sum_data_zip, len(pod_set_tpe))
    bvn_compose, sum_bvn = bvn.matrix_decompose(sum_data_zip, data_matrix_stuff, len(pod_set_tpe), 0.9, num_bvn)
    # if not is_strongly_connected_scipy(sum_bvn):
    #     last_compose = bvn_compose[-1]
    #     ring_compose = sub_ring_edit(last_compose, len(pod_set_tpe))
    #     bvn_compose[-1] = ring_compose
    #     sum_bvn -= last_compose
    #     sum_bvn += ring_compose
    link_matrix_zip = port_allocate(bvn_compose, port, len(pod_set_tpe))
    link_matrix = expand_sub_matrix(link_matrix_zip, pod, pod_set_tpe)
    bool_sum_data = np.where(sum_data_zip > 0, 1, 0)
    bool_link = np.where(link_matrix_zip > 0, 1, 0)
    t_ideal_low = max(np.max(np.sum(sum_data_zip, axis=1)), np.max(np.sum(sum_data_zip, axis=0))) / (
                port * band_per_port)
    flow_no_link = bool_sum_data - bool_link
    no_link_index_row, no_link_index_col = np.where(flow_no_link == 1)
    ideal_data_matrix = t_ideal_low * link_matrix_zip * band_per_port
    delta_compose = ideal_data_matrix - sum_data_zip
    stuff_compose = np.where(bool_link == 1, delta_compose, - np.inf)
    sum_data_edit = copy.deepcopy(sum_data_matrix)
    single_traffic_tpe = [single_traffic[i] for i in job_set_tpe]
    for i in range(len(job_set_tpe)):
        job_index = job_set_tpe[np.argmax(single_traffic_tpe)]
        for j in range(len(no_link_index_row)):
            row, col = no_link_index_row[j], no_link_index_col[j]
            row_stuff, col_stuff = pod_set_tpe[row], pod_set_tpe[col]
            if job_matrix[job_index][row_stuff][col_stuff] == 0:
                continue
            else:
                _, path_zip = max_min_weight_path_dijkstra_k_hops(
                    stuff_compose + 10000000 * np.ones_like(stuff_compose), row, col, 2)
                if len(path_zip) == 0:
                    return -1, np.zeros([pod, pod]), -1
                path = np.zeros(len(path_zip), dtype=int)
                for k in range(len(path_zip)):
                    path[k] = int(pod_set_tpe[path_zip[k]])
                for k in range(len(path) - 1):
                    job_matrix_edit[job_index][path[k]][path[k + 1]] += job_matrix[job_index][row_stuff][col_stuff] + 0
                    sum_data_edit[path[k]][path[k + 1]] += job_matrix[job_index][row_stuff][col_stuff]
                    stuff_compose[path_zip[k]][path_zip[k + 1]] -= job_matrix[job_index][row_stuff][col_stuff]
                job_matrix_edit[job_index][row_stuff][col_stuff] = 0
                sum_data_edit[row_stuff][col_stuff] -= job_matrix[job_index][row_stuff][col_stuff]
        single_traffic_tpe[np.argmax(single_traffic_tpe)] = -1
    link_row, link_col = np.where(link_matrix > 0)
    transmission_times = np.zeros([pod, pod])
    for i in range(len(link_row)):
        transmission_times[link_row[i]][link_col[i]] = (sum_data_edit[link_row[i]][link_col[i]] /
                                                        (band_per_port * link_matrix[link_row[i]][link_col[i]]))
    t_ideal_high = np.max(transmission_times)  # 取最慢的链路
    # print(1)
    if t_ideal_high - t_ideal_low <= t_threshold:
        return job_matrix_edit, sum_data_edit, link_matrix
    job_matrix_output, sum_matrix_output, c = binary_search(t_ideal_low, t_ideal_high, job_matrix, sum_data_matrix,
                                                            link_matrix, band_per_port, single_traffic, t_threshold,
                                                            job_link_index, pod_set_tpe, job_set_tpe)
    if c == 0:
        return job_matrix_edit, sum_data_edit, link_matrix
    return job_matrix_output, sum_matrix_output, link_matrix


# 分组方案
def transport_time(data, link_matrix, band_per_port):
    """

    :param band_per_port:
    :param data:
    :param link_matrix:
    :return:
    """
    flow_row, flow_col = np.where(data > 0)
    t = 0
    for i in range(len(flow_row)):
        row, col = flow_row[i], flow_col[i]
        if link_matrix[row][col] == 0:
            return np.inf
        t_link = data[row][col] / (link_matrix[row][col] * band_per_port)
        if t_link > t:
            t = t_link
    return t


def iteration_time(job_matrix, link_matrix_all, pod, band_per_port, long_group, short_group, long_train, short_train):
    """

    :param job_matrix:
    :param link_matrix_all:
    :param pod:
    :param band_per_port:
    :param long_group:
    :param short_group:
    :param long_train:
    :param short_train:
    :return:
    """
    data_long = np.zeros([pod, pod])
    data_short = np.zeros([pod, pod])
    for i in long_group:
        data_long += job_matrix[i]
    for i in short_group:
        data_short += job_matrix[i]
    long_flow = transport_time(data_long, link_matrix_all, band_per_port)
    short_flow = transport_time(data_short, link_matrix_all, band_per_port)
    long_bool = 0
    short_bool = 0
    # print(long_flow, short_flow)
    if short_train > long_flow:
        short_bool = 1
    if long_train > short_flow:
        long_bool = 1
    return max(short_train, long_flow), max(long_train, short_flow), long_bool, short_bool, long_flow + short_flow


def match_degree_count(job_matrix, long_group, short_group, reverse_group, link_matrix_match, pod, band_per_port):
    """

    :param band_per_port:
    :param pod:
    :param job_matrix:
    :param long_group:
    :param short_group:
    :param reverse_group:
    :param link_matrix_match:
    :return:
    """
    data_long = np.zeros([pod, pod])
    data_short = np.zeros([pod, pod])
    for i in long_group:
        data_long += job_matrix[i]
    for i in short_group:
        data_short += job_matrix[i]
    long_flow = transport_time(data_long, link_matrix_match, band_per_port)
    short_flow = transport_time(data_short, link_matrix_match, band_per_port)
    match_degree_list_long = []
    match_degree_list_short = []
    for i in reverse_group:
        single_flow = transport_time(job_matrix[i], link_matrix_match, band_per_port)
        data_long_add = data_long + job_matrix[i]
        long_flow_add = transport_time(data_long_add, link_matrix_match, band_per_port)
        match_degree_long = (long_flow_add - long_flow) / single_flow
        data_short_add = data_short + job_matrix[i]
        short_flow_add = transport_time(data_short_add, link_matrix_match, band_per_port)
        match_degree_short = (short_flow_add - short_flow) / single_flow
        match_degree_list_long.append(match_degree_long)
        match_degree_list_short.append(match_degree_short)
    return match_degree_list_long, match_degree_list_short


def group(job_matrix, t_train, band_per_port, pod, link_matrix_group, job_set_group):
    """
    分组策略执行
    :param job_set_group:
    :param link_matrix_group:
    :param job_matrix:
    :param t_train:
    :param band_per_port:
    :param pod:
    :return:
    """
    t_train_set = [t_train[i] for i in job_set_group]
    train_sort_index = np.argsort(t_train_set)
    short_group = [job_set_group[train_sort_index[k]] for k in range(len(train_sort_index) - 1)]
    long_group = [job_set_group[train_sort_index[- 1]]]
    t_train_long = t_train_set[train_sort_index[- 1]]
    flag = []
    t_group = []
    while 1:
        if len(short_group) == 0:
            t_train_short = 0
        else:
            t_train_short = max([t_train[i] for i in short_group])
        t1, t2, long_bool, short_bool, _ = iteration_time(job_matrix, link_matrix_group, pod, band_per_port, long_group,
                                                          short_group, t_train_long, t_train_short)
        if long_bool == 1 and short_bool == 0:  # flag = 2
            flag.append(2)
            t_group.append(t1 + t2)
            break
        elif long_bool == 1 and short_bool == 1:  # flag = 3
            flag.append(3)
            t_group.append(t1 + t2)
        elif long_bool == 0 and short_bool == 0:  # flag = 0
            flag.append(0)
            t_group.append(t1 + t2)
        elif long_bool == 0 and short_bool == 1:  # flag = 1
            flag.append(1)
            t_group.append(t1 + t2)
        long_group += [short_group[-1]]
        short_group = [short_group[k] for k in range(len(short_group) - 1)]
    if flag[0] == 2:
        return short_group, long_group
    elif flag[0] == 3:
        if t_group[-1] <= t_group[-2]:
            short_group = [job_set_group[train_sort_index[k]] for k in range(len(job_set_group) - len(flag))]
            long_group = [k for k in job_set_group if k not in short_group]
            return short_group, long_group
        else:
            short_group = [job_set_group[train_sort_index[k]] for k in range(len(job_set_group) - len(flag) + 1)]
            long_group = [k for k in job_set_group if k not in short_group]
            return short_group, long_group
    else:
        len_flag = len(flag)  # flag 长度对应长组长度
        long_group = \
            [job_set_group[train_sort_index[k]] for k in range(len(job_set_group) - len_flag + 1, len(job_set_group))]
        t_train_long = train_sort_index[-1]
        short_group = [job_set_group[train_sort_index[len(job_set_group) - len_flag]]]
        t_train_short = train_time[short_group[0]]
        reverse_group = [job_set_group[train_sort_index[k]] for k in range(len(job_set_group) - len_flag)]
        while 1:
            if len(reverse_group) == 0:
                break
            data_long = np.zeros([pod, pod])
            data_short = np.zeros([pod, pod])
            for i in long_group:
                data_long += job_matrix[i]
            for i in short_group:
                data_short += job_matrix[i]
            long_flow = transport_time(data_long, link_matrix_group, band_per_port)
            short_flow = transport_time(data_short, link_matrix_group, band_per_port)
            match_long, match_short = (
                match_degree_count(job_matrix, long_group, short_group, reverse_group, link_matrix_group, pod,
                                   band_per_port))
            if long_flow >= t_train_short and short_flow < t_train_long:
                job_index = reverse_group[np.argmin(np.array(match_short))]
                short_group.append(job_index)
            elif long_flow < t_train_short and short_flow >= t_train_long:
                job_index = reverse_group[np.argmin(np.array(match_long))]
                long_group.append(job_index)
            else:
                if np.min(np.array(match_short)) < np.min(np.array(match_long)):
                    job_index = reverse_group[np.argmin(np.array(match_short))]
                    short_group.append(job_index)
                else:
                    job_index = reverse_group[np.argmin(np.array(match_long))]
                    long_group.append(job_index)
            reverse_group = [i for i in reverse_group if i != job_index]
        return short_group, long_group


# 主函数
for a in range(2, 3):
    for m in range(1, 6):
        job_number = 100 * m
        result = []
        wb = Workbook()
        ws = wb.active
        for n in reversed(range(4, 5)):
            pod_number = int(8 + 4 * n)
            u = 0
            while u < 5:
                # job1 = Schedule_part.generate_job(job_number)
                job = generate_job.generate_quintuples(job_number, pod_number)
                # if m != 5 and n != 4:
                #     break
                print(a, m, n, u)
                all_job = [i for i in range(0, len(job))]
                # single_link_out, sum_traffic_out = Schedule_part.traffic_count(job1)
                usage = 0.4
                iter_num = 10
                flop = 275
                # train_time = Schedule_part.job_set_train(job1, flop, usage)
                train_time = [job[i][4] for i in range(len(job))]
                b_link = 10000
                port_num = (pod_number - 1) * a
                no_tpe_link = a * (np.ones(pod_number) - np.eye(pod_number))
                # port_num = int(2 * (pod_number - 1))
                # solution_out, undeploy_out, fix_job, unfix_job = Schedule_part.deploy_server(all_job_index, job1, pod_number, 256, 24)
                single_link_out = np.array([job[i][0] for i in range(len(job))])
                sum_traffic_out = np.zeros(len(job))
                for i in range(len(job)):
                    if job[i][1] == 0:
                        sum_traffic_out[i] = job[i][3] * single_link_out[i]
                    else:
                        sum_traffic_out[i] = 2 * (job[i][3] - 1) * single_link_out[i]
                # print(undeploy_out)
                solution_out = generate_job.generate_placement(job, pod_number)
                # print(solution_out)
                # all_job = [i for i in range(job_number) if i not in undeploy_out]
                sum_job_num = 0

                job_set, pod_set = Schedule_part.non_conflict(solution_out, pod_number)
                result_t = []
                result_u = []

                for i in range(len(job_set)):
                    f_job = [j for j in job_set[i] if job[j][1] == 1]
                    uf_job = [j for j in job_set[i] if job[j][1] == 0]
                    pod_set[i] = list(pod_set[i])
                    all_job = [j for j in all_job if j in job_set[i]]
                    # print(all_job)
                    pod_sort = copy.deepcopy(pod_set[i])
                    pod_sort.sort()
                    time1 = time.time()
                    data_matrix, link_job = job_ring(job_set[i], f_job, uf_job, solution_out, single_link_out,
                                                     sum_traffic_out, pod_number, pod_sort, 10)
                    time2 = time.time()
                    # print(time2 - time1)
                    data_matrix, sum_job, link_matrix_end = tpe(job_set[i], data_matrix, link_job, port_num, pod_number,
                                                                10, b_link, single_link_out, pod_sort, 1e-3)
                    if np.max(sum_job) == 0:
                        print("fail")
                        break
                    else:
                        u += 1
                        data_sum = np.zeros([pod_number, pod_number])
                        for j in range(len(data_matrix)):
                            data_sum += data_matrix[j]
                        # time3 = time.time()
                        # print(time3 - time2)
                        g1, g2 = group(data_matrix, train_time, b_link, pod_number, link_matrix_end, job_set[i])
                        # time4 = time.time()
                        print(g1, "\n", g2)
                        # print(time4 - time3)
                        train_g1 = [train_time[i] for i in g1] + [0]
                        train_g2 = [train_time[i] for i in g2] + [0]
                        t_iter = iteration_time(data_matrix, link_matrix_end, pod_number, b_link, g1, g2, max(train_g1),
                                                max(train_g2))
                        # ideal_matrix = b_link * link_matrix_end * (t_iter[0] + t_iter[1])
                        # delta = ideal_matrix - data_matrix
                        data_matrix_not_tpe = np.zeros_like(data_matrix)
                        for ii in range(len(job_set[i])):
                            job_index = job_set[i][ii]

                            worker = solution_out[job_index]
                            if job[job_index][1] == 0:
                                data_matrix_not_tpe[job_index][worker[-1]][worker[0]] += single_link_out[job_index]
                                for k in range(len(worker) - 1):
                                    data_matrix_not_tpe[job_index][worker[k]][worker[k + 1]] += single_link_out[
                                        job_index]
                            if job[job_index][1] == 1:
                                for k in range(len(worker) - 1):
                                    data_matrix_not_tpe[job_index][worker[k]][worker[k + 1]] += single_link_out[
                                        job_index]
                                    data_matrix_not_tpe[job_index][worker[k + 1]][worker[k]] += single_link_out[
                                        job_index]
                        print("main", t_iter[0] + t_iter[1])
                        result_t.append(t_iter[0] + t_iter[1])
                        result_u.append(t_iter[0] + t_iter[1])
                        g3, g4 = group(data_matrix_not_tpe, train_time, b_link, pod_number, no_tpe_link, job_set[i])
                        train_g3 = [train_time[i] for i in g3] + [0]
                        train_g4 = [train_time[i] for i in g4] + [0]
                        t_iter_1 = iteration_time(data_matrix_not_tpe, no_tpe_link, pod_number, b_link, g1, g2,
                                                  max(train_g3),
                                                  max(train_g4))
                        # ideal_matrix = b_link * link_matrix_end * (t_iter[0] + t_iter[1])
                        # delta = ideal_matrix - data_matrix
                        print("no-tpe", t_iter_1[0] + t_iter_1[1])
                        result_t.append(t_iter_1[0] + t_iter_1[1])
                        result_u.append(t_iter_1[0] + t_iter_1[1])
                        # print(data_matrix)
                        # ILP_new.ilp_new(fix_job, unfix_job, train_time, job_number, pod_number, b_link, single_link_out,
                        #                 port_num, solution_out)

                    # cassini 部分
                    conn_matrix = link_matrix_end * b_link
                    # conn_matrix = link_matrix_end * b_link
                    simulator = cassini_schedule.CassiniSimulator(num_servers=pod_number, link_capacity=b_link)
                    # data_matrix_cassini = np.array([np.zeros([pod_number, pod_number]) for _ in range(len(solution_out))])
                    data_sum_cassini = np.zeros([pod_number, pod_number])
                    for ii in range(len(job_set[i])):
                        job_index = job_set[i][ii]

                        # worker = solution_out[job_index]
                        # if job[job_index][1] == 0:
                        #     data_matrix_cassini[job_index][worker[-1]][worker[0]] += single_link_out[job_index]
                        #     for k in range(len(worker) - 1):
                        #         data_matrix_cassini[job_index][worker[k]][worker[k + 1]] += single_link_out[job_index]
                        # if job[job_index][1] == 1:
                        #     for k in range(len(worker) - 1):
                        #         data_matrix_cassini[job_index][worker[k]][worker[k + 1]] += single_link_out[job_index]
                        #         data_matrix_cassini[job_index][worker[k + 1]][worker[k]] += single_link_out[job_index]

                        row, col = np.where(data_matrix[job_index] > 0)
                        t_job = 0
                        for j in range(len(row)):
                            t_link_j = data_matrix[job_index][row[j]][col[j]] / conn_matrix[row[j]][col[j]]
                            if t_link_j > t_job:
                                t_job = t_link_j
                        t_comm = t_job * 2
                        t_comp = train_time[job_index]
                        if t_comm > 0:
                            band_matrix = data_matrix[job_index] / t_comm
                        else:
                            band_matrix = np.zeros([pod_number, pod_number])
                        t_iter = int(1000 * (t_comp + t_comm))
                        t_comp = int(1000 * t_comp)
                        if t_iter == t_comp:
                            t_comp -= 1
                        simulator.add_job(job_index, t_iter, t_comp, band_matrix)

                    avg_times, iter_number = simulator.run_simulation()
                    mean_time = np.mean(avg_times) / 1000
                    max_time = np.max(avg_times) / 1000
                    for c in range(len(job_set[i])):
                        data_sum_cassini += data_matrix[job_set[i][c]] * iter_number[job_set[i][c]]
                    ideal_matrix_cassini = max_time * conn_matrix * (len(pod_set[i]) / pod_number) ** 2
                    delta_cassini = ideal_matrix_cassini - data_sum_cassini
                    print("cassini", mean_time, np.sum(data_sum_cassini) / np.sum(ideal_matrix_cassini))
                    result_t.append(mean_time)
                    result_u.append(max_time)

                    # conn_matrix = (np.ones(pod_number) - np.eye(pod_number)) * b_link
                    network = sjf_schedule.OpticalNetwork(link_matrix_end * b_link, 0)
                    network_las = las_schedule.OpticalNetwork(link_matrix_end * b_link, 0)
                    network_hrrn = hrrn_schedule.OpticalNetwork(link_matrix_end * b_link, 0)
                    # data_matrix_sjf = np.array([np.zeros([pod_number, pod_number]) for _ in range(len(solution_out))])
                    data_sum_sjf = np.zeros([pod_number, pod_number])
                    data_sum_las = np.zeros([pod_number, pod_number])
                    data_sum_hrrn = np.zeros([pod_number, pod_number])
                    data_sum_fcfs = np.zeros([pod_number, pod_number])
                    businesses = []
                    businesses_las = []
                    businesses_hrrn = []
                    for ii in range(len(job_set[i])):
                        job_index = job_set[i][ii]

                        # worker = solution_out[i]
                        # if job[job_index][1] == 0:
                        #     data_matrix[job_index][worker[-1]][worker[0]] += single_link_out[job_index]
                        #     for k in range(len(worker) - 1):
                        #         data_matrix_sjf[job_index][worker[k]][worker[k + 1]] += single_link_out[job_index]
                        # if job[job_index][1] == 1:
                        #     for k in range(len(worker) - 1):
                        #         data_matrix_sjf[job_index][worker[k]][worker[k + 1]] += single_link_out[job_index]
                        #         data_matrix_sjf[job_index][worker[k + 1]][worker[k]] += single_link_out[job_index]

                        biz = sjf_schedule.Business(job_index, train_time[job_index], data_matrix[job_index])
                        biz.calculate_communication_time(link_matrix_end * b_link)
                        biz_las = las_schedule.Business(job_index, train_time[job_index], data_matrix[job_index])
                        biz_las.calculate_communication_time(link_matrix_end * b_link)
                        biz_hrrn = hrrn_schedule.Business(job_index, train_time[job_index], data_matrix[job_index])
                        biz_hrrn.calculate_communication_time(link_matrix_end * b_link)
                        businesses.append(biz)
                        businesses_las.append(biz_las)
                        businesses_hrrn.append(biz_hrrn)
                    avg = sjf_schedule.simulate_sjf_with_starvation(businesses, network, 1000, 4 * m)
                    avg_las = las_schedule.simulate_las_with_starvation(businesses_las, network_las, 1000, 4 * m)
                    avg_hrrn = hrrn_schedule.simulate_hrrn_with_starvation(businesses_hrrn, network_hrrn, 1000, 4 * m)

                    comm_time = np.zeros(len(businesses))
                    avg_fcfs = np.zeros(len(businesses))
                    all_comm = 0
                    for c in range(len(comm_time)):
                        comm_time = businesses[c].communication_time
                        avg_fcfs[c] = all_comm + comm_time + businesses[c].compute_time
                        all_comm += comm_time
                    for c in range(len(businesses)):
                        data_sum_sjf += data_matrix[job_set[i][c]] * (1000 / avg[c])
                    print("SJF", np.max(np.array(avg)), np.mean(np.array(avg)),
                          np.sum(data_sum_sjf) / (1000 * np.sum(link_matrix_end) * b_link))
                    result_t.append(np.mean(np.array(avg)))
                    result_u.append(np.max(np.array(avg)))
                    for c in range(len(businesses)):
                        data_sum_las += data_matrix[job_set[i][c]] * (1000 / avg_las[c])
                    print("LAS", np.max(np.array(avg_las)), np.mean(np.array(avg_las)),
                          np.sum(data_sum_las) / (1000 * np.sum(link_matrix_end) * b_link))
                    result_t.append(np.mean(np.array(avg_las)))
                    result_u.append(np.max(np.array(avg_las)))
                    for c in range(len(businesses)):
                        data_sum_hrrn += data_matrix[job_set[i][c]] * (1000 / avg_hrrn[c])
                    print("HRRN", np.max(np.array(avg_hrrn)), np.mean(np.array(avg_hrrn)),
                          np.sum(data_sum_hrrn) / (1000 * np.sum(link_matrix_end) * b_link))
                    result_t.append(np.mean(np.array(avg_hrrn)))
                    result_u.append(np.max(np.array(avg_hrrn)))
                    for c in range(len(businesses)):
                        data_sum_fcfs += data_matrix[job_set[i][c]]
                    print("FCFS", np.max(avg_fcfs), np.mean(np.array(avg_fcfs)),
                          np.sum(data_sum_fcfs) / (np.max(avg_fcfs) * np.sum(link_matrix_end) * b_link))
                    result_t.append(np.mean(avg_fcfs))
                    result_u.append(np.max(avg_fcfs))

                result.append(result_t + result_u)
            for k in range(len(result)):
                ws.append(result[k])
                wb.save(f"output_{m}_jobs.xlsx")
