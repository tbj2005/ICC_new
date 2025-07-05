import copy
import random
import ILP_new
import LP_relax
import time
from openpyxl import Workbook
import numpy as np


workbook = Workbook()
sheet = workbook.active
sheet.title = "evaluation"
data = [["TPE-PJS-heu", "TPE-PJS-HC", "TPE-JS", "TPE-heu", "PJS", "t1", "t2", "t3", "t4", "t5"]]
for row in data:
    sheet.append(row)


def init_group_old(job_set, sum_traffic, pod_num, solution):
    fa = 0
    fb = 0
    a_init = []
    b_init = []
    ps_index = np.array([job_set[i][0] for i in range(0, len(job_set)) if job_set[i][3] == 1])
    rar_index = np.array([job_set[i][0] for i in range(0, len(job_set)) if job_set[i][3] == 0])
    ep_index = np.array([job_set[i][0] for i in range(0, len(job_set)) if job_set[i][3] == 2])
    ps_size = np.array([sum_traffic[i] for i in range(0, len(job_set)) if job_set[i][3] == 1])
    rar_size = np.array([sum_traffic[i] for i in range(0, len(job_set)) if job_set[i][3] == 0])
    ep_num = np.array([job_set[i][4] for i in range(0, len(job_set)) if job_set[i][3] == 2])
    tba = np.zeros([pod_num, pod_num], dtype=bool)
    tbb = np.zeros([pod_num, pod_num], dtype=bool)
    for i in range(0, len(ps_index)):
        index = np.argmax(ps_size)
        job_index = ps_index[index]
        ps_node = solution[job_index][0][0]
        worker_node = [solution[job_index][j][0][0] for j in range(1, len(solution[job_index]))]
        if fa <= fb:
            a_init.append(job_index)
            fa += sum_traffic[job_index]
            for j in range(0, len(solution[job_index]) - 1):
                tba[ps_node][worker_node[j]] = 1
                tba[worker_node[j]][ps_node] = 1
        else:
            b_init.append(job_index)
            fb += sum_traffic[job_index]
            for j in range(0, len(solution[job_index]) - 1):
                tbb[ps_node][worker_node[j]] = 1
                tbb[worker_node[j]][ps_node] = 1
        ps_size[job_index] = 0
    for i in range(0, len(ep_index)):
        index = np.argmax(ep_num)
        job_index = ep_index[index]
        worker_node = [solution[job_index][j][0][0] for j in range(0, len(solution[job_index]))]
        score_a = 0
        score_b = 0
        for p in worker_node:
            for q in worker_node:
                if p > q and tba[p][q] == 1:
                    score_a += 1
                if p > q and tbb[p][q] == 1:
                    score_b += 1
        if score_a <= score_b:
            a_init.append(job_index)
            fa += sum_traffic[job_index]
            for p in worker_node:
                for q in worker_node:
                    if p > q:
                        tba[p][q] = 1
                        tba[q][p] = 1
        else:
            b_init.append(job_index)
            fb += sum_traffic[job_index]
            for p in worker_node:
                for q in worker_node:
                    if p > q:
                        tbb[p][q] = 1
                        tbb[q][p] = 1
    for i in range(0, len(rar_index)):
        index = np.argmax(rar_size)
        job_index = rar_index[index]
        if fa <= fb:
            a_init.append(job_index)
            fa += sum_traffic[job_index]
        else:
            b_init.append(job_index)
            fb += sum_traffic[job_index]
    return a_init, b_init


def cut_job(sum_traffic, threshold):
    sum_traffic = np.array([sum_traffic[i] for i in range(0, len(sum_traffic))])
    sum_all = sum(sum_traffic)
    sum_main = sum_all * threshold
    sum_traffic_copy = copy.deepcopy(sum_traffic)
    count_traffic = 0
    main_job = []
    for i in range(0, len(sum_traffic_copy)):
        if count_traffic <= sum_main:
            index = np.argmax(sum_traffic_copy)
            main_job.append(index)
            count_traffic += sum_traffic_copy[index]
            sum_traffic_copy[index] = 0
        else:
            return main_job


def init_group_new(job_set, sum_traffic, solution, threshold):
    c = np.zeros([len(job_set), len(job_set)])
    adjoin_a = []
    adjoin_b = []
    for i in range(0, len(job_set)):
        for j in range(i, len(job_set)):
            if job_set[i][3] != 1:
                worker_i = [solution[i][k1][0][0] for k1 in range(0, len(solution[i]))]
            else:
                worker_i = [solution[i][k1][0][0] for k1 in range(1, len(solution[i]))]
            if job_set[j][3] != 1:
                worker_j = [solution[j][k1][0][0] for k1 in range(0, len(solution[j]))]
            else:
                worker_j = [solution[j][k1][0][0] for k1 in range(1, len(solution[j]))]
            score = 0
            for n in worker_i:
                if n in worker_j:
                    score += 1
            c[i][j] = score
            c[j][i] = score
    main_job = cut_job(sum_traffic, threshold)
    min_job = [job_set[i][0] for i in range(0, len(job_set)) if int(job_set[i][0]) not in main_job]
    fa = 0
    fb = 0
    a_init = []
    b_init = []
    for i in main_job:
        if fa <= fb:
            a_init.append(i)
            fa += sum_traffic[i]
        else:
            b_init.append(i)
            fb += sum_traffic[i]
    adjoin_a = copy.deepcopy(min_job)
    adjoin_b = copy.deepcopy(min_job)
    """
    for i in min_job:
        score_a = sum([c[i][j] for j in a_init])
        score_b = sum([c[i][j] for j in b_init])
        if score_a <= score_b:
            a_init.append(i)
        else:
            b_init.append(i)
    """
    score_a = 0
    score_b = 0
    for i in a_init:
        for j in a_init:
            if i > j:
                score_a += c[i][j]
    for i in b_init:
        for j in b_init:
            if i > j:
                score_b += c[i][j]
    while len(min_job) > 0:
        if score_a < score_b:
            adjoin_a = copy.deepcopy(min_job)
            adjoin_weight = []
            for i in adjoin_a:
                adjoin_weight.append(sum([c[i][j] for j in a_init]))
            min_index = np.argmax(adjoin_weight)
            job_index = adjoin_a[min_index]
            a_init.append(job_index)
            score_a += adjoin_weight[min_index]
            fa += sum_traffic[job_index]
        else:
            adjoin_weight = []
            adjoin_b = copy.deepcopy(min_job)
            for i in adjoin_b:
                adjoin_weight.append(sum([c[i][j] for j in b_init]))
            min_index = np.argmax(adjoin_weight)
            job_index = adjoin_b[min_index]
            b_init.append(job_index)
            score_b += adjoin_weight[min_index]
            fb += sum_traffic[job_index]
        min_job = [i for i in min_job if i != job_index]
    print(score_a, score_b, np.sum(c))
    return a_init, b_init, c


def tpe_js(solution, undeploy, job_set, pod_num, unit_gpu, per_oxc_port, b_tor, b_oxc, t_recon, gpu_usage, gpu_flops):
    train_time = [train_server(job_set[i][2], job_set[i][1], gpu_flops, gpu_usage, job_set[i][4] * job_set[i][5]) for i
                  in range(0, len(job_set))]
    if len(undeploy) > 1:
        return -1
    a = [job_set[i][0] for i in range(0, len(job_set))]
    single_link, sum_traffic = traffic_count(job_set)
    t = round_time(a, [], job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port, train_time, t_recon)
    return t


def round_time(a, b, job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port, train_time, t_recon):
    traffic_push_a, tor_push_a, traffic_pull_a, tor_pull_a = traffic_topo(a, job_set, solution, pod_num, single_link)
    traffic_push_b, tor_push_b, traffic_pull_b, tor_pull_b = traffic_topo(b, job_set, solution, pod_num, single_link)
    t_iter_push_a, topo_iter_push_a = oxc_count(traffic_push_a, tor_push_a, per_oxc_port, b_tor, b_oxc)
    t_iter_pull_a, topo_iter_pull_a = oxc_count(traffic_pull_a, tor_pull_a, per_oxc_port, b_tor, b_oxc)
    t_iter_push_b, topo_iter_push_b = oxc_count(traffic_push_b, tor_push_b, per_oxc_port, b_tor, b_oxc)
    t_iter_pull_b, topo_iter_pull_b = oxc_count(traffic_pull_b, tor_pull_b, per_oxc_port, b_tor, b_oxc)
    train_a = max([0] + [train_time[j] for j in a])
    train_a_agg = max([0] + [train_time[j] for j in a if job_set[j][3] != 1])
    train_b = max([0] + [train_time[j] for j in b])
    train_b_agg = max([0] + [train_time[j] for j in b if job_set[j][3] != 1])
    t_recon_1 = adjoin_topo(topo_iter_push_a, topo_iter_push_b, per_oxc_port) * t_recon
    t_recon_2 = adjoin_topo(topo_iter_push_b, topo_iter_pull_a, per_oxc_port) * t_recon
    t_recon_3 = adjoin_topo(topo_iter_pull_a, topo_iter_pull_b, per_oxc_port) * t_recon
    t_recon_4 = adjoin_topo(topo_iter_pull_b, topo_iter_push_a, per_oxc_port) * t_recon
    t_1 = max(train_a, t_iter_pull_b + t_recon_4)
    t_2 = max(train_b, t_iter_push_a + t_recon_1)
    t_3 = max(train_a_agg, t_iter_push_b + t_recon_2)
    t_4 = max(train_b_agg, t_iter_pull_a + t_recon_3)
    return t_1 + t_2 + t_3 + t_4


def mountain_climb(solution, undeploy, job_set, pod_num, unit_gpu, per_oxc_port, b_tor, b_oxc, t_recon, gpu_usage,
                   gpu_flops, threshold, iter_lim, change_rate):
    """
    爬山法迭代最优解
    :param undeploy:
    :param solution:
    :param change_rate:
    :param iter_lim:
    :param threshold:
    :param gpu_flops:
    :param gpu_usage:
    :param t_recon:
    :param b_oxc:
    :param b_tor:
    :param per_oxc_port:
    :param unit_gpu:
    :param pod_num:
    :param job_set: 业务六元组集合（业务索引，业务参数量，batch size，通信模式， 并行度， 单个并行单元的 gpu 数目， seq length）,
    通信模式分 ring, ps 两类，分别使用0， 1表示
    :return:
    """
    list_t = []
    single_link, sum_traffic = traffic_count(job_set)
    train_time = [train_server(job_set[i][2], job_set[i][1], gpu_flops, gpu_usage, job_set[i][4] * job_set[i][5]) for i
                  in range(0, len(job_set))]
    if len(undeploy) > 0:
        return -1, -1, -1, -1
    a, b, c = init_group_new(job_set, sum_traffic, solution, threshold)
    print(a, b)
    t_iter = round_time(a, b, job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port, train_time, t_recon)
    end_4 = time.time()
    t_testbench = t_iter + 0
    list_t.append(t_testbench)
    k = 0
    flag = 0
    c_queue = np.zeros(len(job_set))
    for i in range(0, len(c_queue)):
        if i in a:
            c_queue[i] = sum([c[i][j] for j in b])
        else:
            c_queue[i] = sum([c[i][j] for j in a])
    change_num = int(len(job_set) * change_rate)
    while 1:
        # print("iteration" + str(k))
        if k == iter_lim:
            print(a, b)
            return t_iter, t_testbench, (t_testbench - t_iter) / t_testbench, end_4
        t_iter_list = np.zeros(len(a) + len(b))
        k += 1
        c_queue_copy = copy.deepcopy(c_queue)
        c_change = []
        for i in range(0, change_num):
            index = np.argmin(c_queue_copy)
            c_change.append(index)
            c_queue_copy[index] = np.Inf
        """
        if flag == len(a + b):
            return t_iter
        """
        for i in c_change:
            ai = copy.deepcopy(a)
            bi = copy.deepcopy(b)
            if i in a:
                ai = [j for j in a if j != i]
                bi += [i]
            else:
                bi = [j for j in b if j != i]
                ai += [i]
            t_iter_list[i] = round_time(ai, bi, job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port,
                                        train_time, t_recon)
            # print(i, t_iter_list[i])
        # print(t_iter)
        """
        if min(t_iter_list) == np.inf:
            b += [a[0]]
            a = [j for j in a if j != a[0]]
        """
        if np.min(t_iter_list) > t_iter:
            print(a, b)
            return t_iter, t_testbench, (t_testbench - t_iter) / t_testbench, end_4
        else:
            index = np.argmin(t_iter_list)
            if index in a:
                a = [j for j in a if j != index]
                b += [index]
            else:
                b = [j for j in b if j != index]
                a += [index]
            t_iter_list = np.array([i for i in t_iter_list if i != 0])
            t_iter = min(t_iter_list)
            flag_l = 0
            if len(list_t) > 3:
                list_t_copy = [list_t[j] for j in range(len(list_t) - 3, len(list_t))]
                for j in range(1, len(list_t_copy)):
                    list_t_copy[j] -= list_t_copy[0]
                for j in range(1, len(list_t_copy)):
                    if list_t_copy[j] != 0:
                        flag_l = 1
                if flag_l == 0:
                    print(a, b)
                    return t_iter, t_testbench, (t_testbench - t_iter) / t_testbench, end_4
            list_t.append(t_iter)


def round_time_pjs(a, b, job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port, train_time, t_recon):
    traffic_push_a, tor_push_a, traffic_pull_a, tor_pull_a = traffic_topo(a, job_set, solution, pod_num, single_link)
    traffic_push_b, tor_push_b, traffic_pull_b, tor_pull_b = traffic_topo(b, job_set, solution, pod_num, single_link)
    t_push_a = time_count(b_oxc, traffic_push_a, pod_num, tor_push_a, b_tor)
    t_pull_a = time_count(b_oxc, traffic_pull_a, pod_num, tor_pull_a, b_tor)
    t_push_b = time_count(b_oxc, traffic_push_b, pod_num, tor_push_b, b_tor)
    t_pull_b = time_count(b_oxc, traffic_pull_b, pod_num, tor_pull_b, b_tor)
    train_a = max([train_time[i] for i in a] + [0])
    train_b = max([train_time[i] for i in b] + [0])
    train_a_agg = max([train_time[i] for i in a if job_set[i][3] != 1] + [0])
    train_b_agg = max([train_time[i] for i in b if job_set[i][3] != 1] + [0])
    t_1 = max(train_a, t_pull_b)
    t_2 = max(train_b, t_push_a)
    t_3 = max(train_a_agg, t_push_b)
    t_4 = max(train_b_agg, t_pull_a)
    return t_1 + t_2 + t_3 + t_4


def mountain_pjs(solution, undeploy, job_set, pod_num, unit_gpu, per_oxc_port, b_tor, b_oxc, t_recon, gpu_usage,
                 gpu_flops, threshold, iter_lim, change_rate):
    list_t = []
    train_time = [train_server(job_set[i][2], job_set[i][1], gpu_flops, gpu_usage, job_set[i][4] * job_set[i][5]) for i
                  in range(0, len(job_set))]
    single_link, sum_traffic = traffic_count(job_set)
    if len(undeploy) > 0:
        return -1, -1, -1, -1
    a, b, c = init_group_new(job_set, sum_traffic, solution, threshold)
    t_iter = round_time_pjs(a, b, job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port, train_time,
                            t_recon)
    end_5 = time.time()
    t_testbench = t_iter + 0
    list_t.append(t_testbench)
    k = 0
    flag = 0
    c_queue = np.zeros(len(job_set))
    for i in range(0, len(c_queue)):
        if i in a:
            c_queue[i] = sum([c[i][j] for j in b])
        else:
            c_queue[i] = sum([c[i][j] for j in a])
    change_num = int(len(job_set) * change_rate)
    while 1:
        # print("iteration" + str(k))
        if k == iter_lim:
            return t_iter, t_testbench, (t_testbench - t_iter) / t_testbench, end_5
        t_iter_list = np.zeros(len(a) + len(b))
        k += 1
        c_queue_copy = copy.deepcopy(c_queue)
        c_change = []
        for i in range(0, change_num):
            index = np.argmin(c_queue_copy)
            c_change.append(index)
            c_queue_copy[index] = np.Inf
        """
        if flag == len(a + b):
            return t_iter
        """
        for i in c_change:
            ai = copy.deepcopy(a)
            bi = copy.deepcopy(b)
            if i in a:
                ai = [j for j in a if j != i]
                bi += [i]
            else:
                bi = [j for j in b if j != i]
                ai += [i]
            t_iter_list[i] = round_time_pjs(ai, bi, job_set, b_tor, b_oxc, solution, pod_num, single_link, per_oxc_port,
                                            train_time, t_recon)
            # print(i, t_iter_list[i])
        # print(t_iter)
        t_iter_list = np.array([i for i in t_iter_list if i != 0])
        if np.min(t_iter_list) > t_iter:
            return t_iter, t_testbench, (t_testbench - t_iter) / t_testbench, end_5
        else:
            index = np.argmin(t_iter_list)
            if index in a:
                a = [j for j in a if j != index]
                b += [index]
            else:
                b = [j for j in b if j != index]
                a += [index]
            t_iter = min(t_iter_list)
            flag_l = 0
            if len(list_t) > 3:
                list_t_copy = [list_t[j] for j in range(len(list_t) - 3, len(list_t))]
                for j in range(1, len(list_t_copy)):
                    list_t_copy[j] -= list_t_copy[0]
                for j in range(1, len(list_t_copy)):
                    if list_t_copy[j] != 0:
                        flag_l = 1
                if flag_l == 0:
                    return t_iter, t_testbench, (t_testbench - t_iter) / t_testbench, end_5
            list_t.append(t_iter)


def time_count(b_oxc, traffic, pod_num, tor_traffic, b_tor):
    t = 0
    for i in range(0, pod_num):
        for j in range(0, pod_num):
            if traffic[i][j] > 0:
                if traffic[i][j] / b_oxc > t:
                    t = traffic[i][j] / b_oxc
    for i in range(0, pod_num):
        if tor_traffic[i] > 0:
            if tor_traffic[i] / b_tor > t:
                t = tor_traffic[i] / b_tor
    return t


def adjoin_topo(topo_1, topo_2, oxc_port):
    topo_joint = np.zeros_like(topo_1)
    pod_num = topo_joint.shape[1]
    for i in range(0, pod_num):
        for j in range(0, pod_num):
            topo_joint[i][j] = max(topo_1[i][j] + topo_1[j][i], topo_2[i][j] + topo_2[j][i])
    use_port = np.sum(topo_joint, axis=1)
    for i in range(0, pod_num):
        if use_port[i] > oxc_port:
            return 1
    return 0


def oxc_count(oxc_traffic, tor_traffic, oxc_port, tor_bandwidth, link_bandwidth):
    pod_num = len(tor_traffic)
    oxc_topo = np.zeros([pod_num, pod_num])
    degree_pod = np.zeros(pod_num)
    t_test = np.zeros([pod_num, pod_num])
    t_tor = np.zeros(pod_num)
    for i in range(0, pod_num):
        for j in range(0, pod_num):
            if oxc_traffic[i][j] > 0:
                oxc_topo[i][j] = 1
                t_test[i][j] = oxc_traffic[i][j] / (oxc_topo[i][j] * link_bandwidth)
                degree_pod[i] += 1
                degree_pod[j] += 1
    for i in range(0, pod_num):
        t_tor[i] = tor_traffic[i] / tor_bandwidth
    degree = np.sum(oxc_topo, axis=0) + np.sum(oxc_topo, axis=1)
    # print(degree)
    reverse_degree = np.array([oxc_port - degree[i] for i in range(0, pod_num)])
    if np.count_nonzero(reverse_degree[reverse_degree < 0]) > 0:
        return -1, np.zeros([pod_num, pod_num])
    max_index_tor = np.argmax(t_tor)
    while 1:
        max_index = np.argmax(t_test)
        max_row_oxc = int(max_index / pod_num)
        max_col_oxc = int(max_index % pod_num)
        if t_test[max_row_oxc][max_col_oxc] <= t_tor[max_index_tor]:
            break
        else:
            if degree_pod[max_row_oxc] < oxc_port and degree_pod[max_col_oxc] < oxc_port:
                degree_pod[max_row_oxc] += 1
                degree_pod[max_col_oxc] += 1
                oxc_topo[max_row_oxc][max_col_oxc] += 1
                t_test[max_row_oxc][max_col_oxc] = (oxc_traffic[max_row_oxc][max_col_oxc] /
                                                    (oxc_topo[max_row_oxc][max_col_oxc] * link_bandwidth))
            else:
                break
    t_min = max(t_tor[max_index_tor], np.max(t_test))
    return t_min, oxc_topo


def traffic_topo(group, job_set, local_solution, pod_num, traffic_single_link):
    """
    计算每组的 OXC 拓扑，并求解对应的单迭代时长
    :param traffic_single_link:
    :param pod_num:
    :param local_solution:
    :param group: a 组业务索引
    :param job_set: 业务六元组集合
    :return:
    """
    ring_a = []
    all_job = np.array([job_set[i][0] for i in range(0, len(job_set))])
    traffic_matrix_push = np.zeros([pod_num, pod_num])
    tor_traffic_push = np.zeros(pod_num)
    traffic_matrix_pull = np.zeros([pod_num, pod_num])
    tor_traffic_pull = np.zeros(pod_num)
    traffic_matrix_push_all = np.zeros([pod_num, pod_num])
    traffic_matrix_pull_all = np.zeros([pod_num, pod_num])
    for i in all_job:
        if job_set[i][3] == 0:
            continue
        elif job_set[i][3] == 1:
            ps_node = local_solution[i][0][0]
            worker_node = [local_solution[i][j][0][0] for j in range(1, len(local_solution[i]))]
            for j in worker_node:
                traffic_matrix_push_all[j][ps_node] += traffic_single_link[i]
                traffic_matrix_pull_all[ps_node][j] += traffic_single_link[i]
        elif job_set[i][3] == 2:
            worker_node = [local_solution[i][j][0][0] for j in range(0, len(local_solution[i]))]
            for p in worker_node:
                for q in worker_node:
                    if p != q:
                        traffic_matrix_push_all[p][q] += traffic_single_link[i]
                        traffic_matrix_pull_all[p][q] += traffic_single_link[i]
    # link_bool_a_1 = np.zeros([pod_num, pod_num])
    for n in range(0, 2):
        link_bool_a_1 = np.zeros([pod_num, pod_num])
        for i in group:
            if len(local_solution[i]) == 0:
                continue
            single_oxc_traffic = np.zeros([pod_num, pod_num])
            if job_set[i][3] == 0:
                ring_a.append(i)
            elif job_set[i][3] == 1:
                local_job = local_solution[i]
                ps_node = local_job[0][0]
                for j in range(1, job_set[i][4] + 1):
                    unit_job = local_job[j]
                    rank_node, rank_node_gpu = unit_job[0][0], unit_job[0][1]
                    tor_traffic_push[rank_node] += rank_node_gpu * traffic_single_link[i] / job_set[i][5]
                    if rank_node != ps_node:
                        if n == 0:
                            single_oxc_traffic[rank_node][ps_node] += traffic_single_link[i]
                        else:
                            single_oxc_traffic[ps_node][rank_node] += traffic_single_link[i]
                        # link_bool_a_1[rank_node][ps_node] = 1
                    for k in range(1, len(unit_job)):
                        sub_node, sub_node_gpu = unit_job[k][0], unit_job[k][1]
                        # link_bool_a_1[rank_node][sub_node] = 1
                        single_oxc_traffic[rank_node][sub_node] += (
                                sub_node_gpu * traffic_single_link[i] / job_set[i][5])
                        tor_traffic_push[sub_node] += sub_node_gpu * traffic_single_link[i] / job_set[i][5]
            elif job_set[i][3] == 2:
                local_job = local_solution[i]
                for j in range(0, job_set[i][4]):
                    unit_job = local_job[j]
                    for k in range(0, len(unit_job)):
                        node, node_gpu = unit_job[k][0], unit_job[k][1]
                        tor_traffic_push[node] += (
                                node_gpu * traffic_single_link[i] * (job_set[i][4] - 1) / job_set[i][5])
                        if k > 0:
                            single_oxc_traffic[node][unit_job[0][0]] += (
                                    node_gpu * traffic_single_link[i] * (job_set[i][4] - 1) / job_set[i][5])
                for p in range(0, job_set[i][4]):
                    for q in range(0, job_set[i][4]):
                        if p != q:
                            if local_job[p][0][0] != local_job[q][0][0]:
                                single_oxc_traffic[local_job[p][0][0]][local_job[q][0][0]] += traffic_single_link[i]
            if n == 0:
                traffic_matrix_push += single_oxc_traffic
            else:
                traffic_matrix_pull += single_oxc_traffic
            """
            for i in range(0, pod_num):
                for j in range(0, pod_num):
                    if traffic_matrix[i][j] > 0:
                        link_bool_a_1[i][j] = 1
            """
        if n == 0:
            link_bool_a_1 = (traffic_matrix_push_all > 0).astype(int)
        else:
            link_bool_a_1 = (traffic_matrix_pull_all > 0).astype(int)
        for i in ring_a:
            ring_oxc_traffic, ring_tor_traffic = (
                connect_link_ring(link_bool_a_1, traffic_single_link[i], local_solution[i], job_set[i]))
            link_bool_a_1 = (ring_oxc_traffic > 0).astype(int)
            if n == 0:
                traffic_matrix_push += ring_oxc_traffic
                traffic_matrix_push_all += ring_tor_traffic
                tor_traffic_push += ring_tor_traffic
            else:
                traffic_matrix_pull += ring_oxc_traffic
                traffic_matrix_pull_all += ring_oxc_traffic
                tor_traffic_pull += ring_tor_traffic
    return traffic_matrix_push, tor_traffic_push, traffic_matrix_pull, tor_traffic_pull


def deploy_server(group, job_set, pod, pod_gpu, seg_pod_num):
    local_solution = np.empty(len(group), dtype=object)
    reverse_gpu = np.array([pod_gpu for _ in range(0, pod)])
    seg_num = int(pod / seg_pod_num)
    seg_pod_reserve = np.array([seg_pod_num * pod_gpu for _ in range(0, seg_num)])
    unit_gpu_num = np.zeros(len(group))
    fj = [0 for _ in range(len(job_set))]
    ufj = [0 for _ in range(len(job_set))]
    for i1 in range(0, len(job_set)):
        unit_gpu_num[i1] = job_set[i1][5]
    undeploy = []
    for i1 in range(0, len(job_set)):
        local_solution[i1] = [[], [0 for _ in range(pod)]]
        if job_set[i1][3] != 1:
            ufj[i1] = 1
            available_seg = [n for n in range(0, seg_num) if seg_pod_reserve[n] >= unit_gpu_num[i1] * job_set[i1][4]]
            if len(available_seg) == 0:
                local_solution[i1][0] = -2
                undeploy.append(i1)
                continue
            seg_index = random.choice(available_seg)
            worker_node = []
            for j in range(0, job_set[i1][4]):
                seg_reserve = reverse_gpu[int(seg_index * seg_pod_num): int(seg_index * seg_pod_num + seg_pod_num)]
                max_index = np.argmax(seg_reserve) + seg_index * seg_pod_num
                if reverse_gpu[max_index] < unit_gpu_num[i1]:
                    for i in range(0, len(worker_node)):
                        reverse_gpu[worker_node[i][0]] += unit_gpu_num[i1]
                        seg_pod_reserve[seg_index] += unit_gpu_num[i1]
                    undeploy.append(i1)
                    worker_node = []
                    break
                seg_reserve[max_index - seg_index * seg_pod_num] -= unit_gpu_num[i1]
                worker_node.append((max_index, unit_gpu_num[i1]))
            if i1 in undeploy:
                local_solution[i1][0] = -2
                continue
            local_solution[i1][0] = -1
            for j in range(0, len(worker_node)):
                local_solution[i1][1][worker_node[j][0]] += worker_node[j][1]
        else:
            fj[i1] = 1
            available_seg = [n for n in range(0, seg_num) if
                             seg_pod_reserve[n] >= unit_gpu_num[i1] * job_set[i1][4] + 1]
            if len(available_seg) == 0:
                local_solution[i1][0] = -2
                undeploy.append(i1)
                continue
            seg_index = random.choice(available_seg)
            worker_node = []
            # for n in range(0, seg_pod_num):
            seg_reserve_gpu = reverse_gpu[int(seg_index * seg_pod_num): int(seg_index * seg_pod_num) + seg_pod_num]
            in_seg_index = np.argmax(seg_reserve_gpu)
            local_solution[i1][0] = int(seg_index * seg_pod_num + in_seg_index)
            reverse_gpu[int(seg_index * seg_pod_num + in_seg_index)] -= 1
            worker_node.append((seg_index * seg_pod_num + in_seg_index, 1))
            # if reverse_gpu[int(seg_index * seg_pod_num + n)] >= 1:
            #     worker_node.append((seg_index * seg_pod_num + n, 1))
            #     reverse_gpu[int(seg_index * seg_pod_num + n)] -= 1
            #     local_solution[i1][0] = int(seg_index * seg_pod_num + n)
            #     break
            for j in range(0, job_set[i1][4]):
                seg_reserve = reverse_gpu[int(seg_index * seg_pod_num): int(seg_index * seg_pod_num + seg_pod_num)]
                max_index = np.argmax(seg_reserve) + seg_index * seg_pod_num
                if reverse_gpu[max_index] < unit_gpu_num[i1]:
                    for i in range(0, len(worker_node)):
                        if i == 0:
                            reverse_gpu[worker_node[i][0]] += 1
                            seg_pod_reserve[seg_index] += 1
                            continue
                        reverse_gpu[worker_node[i][0]] += unit_gpu_num[i1]
                        seg_pod_reserve[seg_index] += unit_gpu_num[i1]
                        undeploy.append(i1)
                    worker_node = []
                    break
                seg_reserve[max_index - seg_index * seg_pod_num] -= unit_gpu_num[i1]
                worker_node.append((max_index, unit_gpu_num[i1]))
            if i1 in undeploy:
                local_solution[i1][0] = -2
                continue
            for j in range(1, len(worker_node)):
                local_solution[i1][1][worker_node[j][0]] += worker_node[j][1]
    return local_solution, undeploy, fj, ufj


def connect_link_ring(link_matrix, single_traffic, local_job, job_info):
    """

    :param job_info:
    :param local_job:
    :param single_traffic:
    :param link_matrix:
    :return:
    """
    rep_link = np.zeros_like(link_matrix)
    pod_num = link_matrix.shape[0]
    node_deploy = []
    for i in range(0, len(local_job)):
        node_deploy.append(local_job[i][0][0])
    node_deploy = list(set(node_deploy))
    for i in range(0, pod_num):
        for j in range(0, pod_num):
            if link_matrix[i][j] == 1:
                if i in node_deploy and j in node_deploy:
                    rep_link[i][j] = 1
    out_degree = np.sum(rep_link, axis=1)
    in_degree = np.sum(rep_link, axis=0)
    ring = []
    for i in node_deploy:
        while in_degree[i] > 1:
            adjoin_node = [j for j in range(0, pod_num) if rep_link[j][i] == 1]
            adjoin_degree = np.array([out_degree[j] for j in adjoin_node])
            max_index = adjoin_node[np.argmax(adjoin_degree)]
            rep_link[max_index][i] = 0
            in_degree[i] -= 1
            out_degree[max_index] -= 1
        while out_degree[i] > 1:
            adjoin_node = [j for j in range(0, pod_num) if rep_link[i][j] == 1]
            adjoin_degree = np.array([in_degree[j] for j in adjoin_node])
            max_index = adjoin_node[np.argmax(adjoin_degree)]
            rep_link[i][max_index] = 0
            out_degree[i] -= 1
            in_degree[max_index] -= 1
    test_node = copy.deepcopy(node_deploy)
    while len(test_node) > 0:
        sub_ring = [test_node[0]]
        test_node = [i for i in test_node if i != test_node[0]]
        while 1:
            adjoin = [i for i in range(0, pod_num) if rep_link[sub_ring[-1]][i] == 1]
            if len(adjoin) == 0:
                break
            else:
                next_node = adjoin[0]
                if next_node == sub_ring[0]:
                    break
                test_node = [i for i in test_node if i != next_node]
                sub_ring.append(next_node)
        ring += sub_ring
    oxc_traffic = np.zeros([pod_num, pod_num])
    tor_traffic = np.zeros(pod_num)
    for i in range(0, len(ring) - 1):
        oxc_traffic[ring[i]][ring[i + 1]] = single_traffic
    oxc_traffic[-1][0] = single_traffic
    for i in range(0, len(local_job)):
        unit_job = local_job[i]
        for j in range(0, len(unit_job)):
            node, node_gpu = unit_job[j][0], unit_job[j][1]
            tor_traffic[node] = node_gpu * single_traffic / job_info[5]
            if j >= 1:
                oxc_traffic[node][unit_job[0][0]] = node_gpu * single_traffic / job_info[5]
    """
    degree_after = np.sum(link_single_matrix + link_matrix, axis= 0)
    if sum([1 for i in range(0, pod_num) if degree_after[i] > oxc_port]) > 0:
        link_single_matrix = np.zeros([pod_num, pod_num])
    link_matrix += link_single_matrix
    """
    return oxc_traffic, tor_traffic


def count_train_time(t_list, gpu_available):
    """
    计算用满所有 GPU 的平均时间和分配方式
    :param t_list: 单 GPU 训练时间队列
    :param gpu_available: 可用 GPU 数目
    :return:
    """
    t_train = sum(t_list) / gpu_available
    gpu_assign = [t_list[i] / t_train for i in range(0, len(t_list))]
    return t_train, gpu_assign


def train_server(batch_size, seq_len, model_parameter, flops, usage, gpu_num):
    """
    业务单 GPU 训练用时
    :param gpu_num:
    :param batch_size:
    :param seq_len:
    :param model_parameter: 业务参数量
    :param flops: GPU 峰值 flops
    :param usage: GPU 利用率
    :return:
    """
    return 0.008 * batch_size * seq_len * model_parameter / (flops * usage * gpu_num)


def job_set_train(job_set, flops, usage):
    train_time_job = np.zeros(len(job_set))
    for i in range(len(job_set)):
        train_time_job[i] = (
            train_server(job_set[i][2], job_set[i][6], job_set[i][1], flops, usage, job_set[i][4] * job_set[i][5]))
    return train_time_job


def traffic_count(job_set):
    """
    根据业务种类，参数量和 batch size 计算通信流量
    :param job_set:
    :return:
    """
    traffic = []
    sum_output_traffic = []
    for i in range(0, len(job_set)):
        if job_set[i][3] == 0:
            dp = job_set[i][4]
            traffic.append(2 * (dp - 1) * job_set[i][1] / dp)
            sum_output_traffic.append(2 * (dp - 1) * job_set[i][1])
        if job_set[i][3] == 1:
            dp = job_set[i][4]
            traffic.append(job_set[i][1])
            sum_output_traffic.append(2 * dp * job_set[i][1])
        # if job_set[i][3] == 2:
        #     ep = job_set[i][4]
        #     traffic.append((job_set[i][2] * 4) * 1e-6 / (ep * ep))
        #     sum_output_traffic.append((ep - 1) * job_set[i][2] * 1e-6 / ep)
    return traffic, sum_output_traffic


def generate_job(job_num):
    job_set = []
    # job_set2 = []
    # job_set3 = []
    # job_set4 = []
    # job_set5 = []
    for i in range(0, job_num):
        job_size = np.random.randint(0, 4)
        para = [2.47, 6.43, 16.07, 21.35]
        job_para = para[job_size]
        batch_size = np.random.choice(np.array([16, 32, 64, 128]), p=np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4]))
        seq_len = np.random.choice(np.array([64, 128, 256, 512, 1024]), p=np.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]))
        # context = 8
        # job_para = np.random.choice(np.array([1, 3, 8, 70]), p=np.array([0.25, 0.25, 0.25]))
        job_type = np.random.choice(np.array([0, 1]), p=np.array([1 / 2, 1 / 2]))
        parallel = np.random.choice(np.array([2, 4, 6]), p=np.array([1 / 3, 1 / 3, 1 / 3]))
        # token_per_batch =
        # np.random.choice(np.array([128, 256, 512, 1024, 2048]), p=np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        unit_gpu = np.random.randint(10, 20)
        job_default = [i, job_para, batch_size, job_type, parallel, unit_gpu, seq_len]
        # job_context_2 = [i, job_para, 2, job_type, parallel, unit_gpu]
        # job_context_3 = [i, job_para, 4, job_type, parallel, unit_gpu]
        # job_context_4 = [i, job_para, 8, job_type, parallel, unit_gpu]
        # job_context_5 = [i, job_para, 16, job_type, parallel, unit_gpu]

        job_set.append(job_default)
        # job_set2.append(job_context_2)
        # job_set3.append(job_context_3)
        # job_set4.append(job_context_4)
        # job_set5.append(job_context_5)
    return job_set


def transform(job_set, solution, pod_num):
    worker_node = []
    ps_n = []
    rar = np.zeros(len(solution))
    ps = np.zeros(len(solution))
    ep = np.zeros(len(solution))
    for i in range(0, len(solution)):
        if job_set[i][3] == 0:
            rar[i] = 1
        elif job_set[i][3] == 1:
            ps[i] = 1
        elif job_set[i][3] == 2:
            ep[i] = 1
    for i in range(0, len(solution)):
        worker_job = np.zeros(pod_num)
        ps_job = np.zeros(pod_num)
        if job_set[i][3] == 1:
            ps_job[solution[i][0][0]] += 1
            for j in range(1, len(solution[i])):
                worker_job[solution[i][j][0][0]] += 1
        else:
            for j in range(0, len(solution[i])):
                worker_job[solution[i][j][0][0]] += 1
        worker_node.append(worker_job)
        ps_n.append(ps_job)
    return worker_node, ps_n, rar, ps, ep


def non_conflict(local_solution, pod_number):
    non_conflict_set = []
    pod_conflict_set = []
    job_number = len(local_solution)
    for i in range(job_number):
        len_set = len(non_conflict_set)
        worker_set = set(local_solution[i])
        if len(worker_set) == 0:
            continue
        if len_set == 0:
            non_conflict_set.append([i])
            pod_conflict_set.append(worker_set)
        else:
            flag_l = []
            for l in range(len_set):
                if len(worker_set & pod_conflict_set[l]) == 0:
                    continue
                else:
                    flag_l.append(l)
            if len(flag_l) == 0:
                non_conflict_set.append([i])
                pod_conflict_set.append(worker_set)
            else:
                delete_set = [non_conflict_set[k] for k in flag_l]
                delete_set_pod = [pod_conflict_set[k] for k in flag_l]
                flag_reverse = flag_l[::-1]
                for k in flag_reverse:
                    del non_conflict_set[k]
                    del pod_conflict_set[k]
                non_conflict_set.append([])
                pod_conflict_set.append(set())
                for k in range(len(delete_set)):
                    non_conflict_set[-1] += delete_set[k]
                    pod_conflict_set[-1] = pod_conflict_set[-1] | delete_set_pod[k]
                non_conflict_set[-1].append(i)
                pod_conflict_set[-1] = pod_conflict_set[-1] | worker_set
    return non_conflict_set, pod_conflict_set


def local_data_revert(local_solution, data_per_worker, change_boolean, pod_num):
    if change_boolean == 0:
        data_matrix = np.array([np.zeros([pod_num, pod_num]) for _ in range(len(local_solution))])
        for i in range(0, len(local_solution)):
            if local_solution[i][0] == -1:
                worker_job = [x for x in range(len(local_solution[i][1])) if local_solution[i][1][x] > 0]
                if len(worker_job) == 1:
                    continue
                for u1 in range(0, len(worker_job)):
                    if u1 == len(worker_job) - 1:
                        data_matrix[i, worker_job[u1], worker_job[0]] = data_per_worker[i]
                    else:
                        data_matrix[i, worker_job[u1], worker_job[u1] + 1] = data_per_worker[i]
                continue
            for x in range(0, pod_number):
                if x != local_solution[i][0]:
                    data_matrix[i, x, local_solution[i][0]] = data_per_worker[i] * local_solution[i][1][x]
                    data_matrix[i, local_solution[i][0], x] = data_per_worker[i] * local_solution[i][1][x]
                    # 通过放置约束固定流量矩阵
                if x == local_solution[i][0]:
                    data_matrix[i, x, local_solution[i][0]] = 0
                    data_matrix[i, local_solution[i][0], x] = 0
        return data_matrix


def count_group_time(aoi_link, group, data_matrix_each_job, pod_num, change_boolean, link_bandwidth):
    if change_boolean == 0:
        all_data_group = sum(np.array([data_matrix_each_job[i] for i in group]))
        t_link = np.zeros([pod_num, pod_num])
        for up in range(pod_num):
            for vp in range(pod_num):
                if all_data_group[up][vp] > 0:
                    if aoi_link[up][vp] == 0:
                        return -1
                    else:
                        t_link[up][vp] = all_data_group[up][vp] / (aoi_link[up][vp] * link_bandwidth)
        return np.max(t_link)


def match_degree(long, short, train_long, train_short, train_index, job_index, boolean_change, data_matrix_each_job, pod_num, link_bandwidth, t_old, local_solution, port):
    if boolean_change == 0:
        long_add = long + [job_index]
        short_add = short + [job_index]
        aoi_link_add_long = topo_each_group(data_matrix_each_job, pod_num, 0, long_add, short, train_long, train_short, local_solution, port, link_bandwidth)
        aoi_link_add_short = topo_each_group(data_matrix_each_job, pod_num, 0, long, short_add, train_long, train_short, local_solution, port, link_bandwidth)
        aoi_link_single = topo_each_group(data_matrix_each_job, pod_num,  0, [job_index], [], train_index, [], local_solution, port, link_bandwidth)
        t_long_add_long = count_group_time(aoi_link_add_long, long_add, data_matrix_each_job, pod_num, boolean_change, link_bandwidth)
        t_long_add_short = count_group_time(aoi_link_add_long, short, data_matrix_each_job, pod_num, boolean_change, link_bandwidth)
        return (t_after - t_group) / t_single


def job_allocate_fix(long, short, reverse, data_matrix_each_job, pod_num, link_bandwidth, train_long, train_short, local_solution, port):
    while len(reverse) > 0:
        reverse_data = np.array([np.sum(data_matrix_each_job[i]) for i in reverse])
        aoi_link = (
            topo_each_group(data_matrix_each_job, pod_num, 0, long, short, train_long, train_short, local_solution, port, link_bandwidth))
        long_time = count_group_time(aoi_link, long, data_matrix_each_job, pod_num, 0, link_bandwidth)
        short_time = count_group_time(aoi_link, short, data_matrix_each_job, pod_num, 0, link_bandwidth)
        if long_time >= train_short and short_time >= train_long:
            max_data_index = reverse[np.argmax(reverse_data)]
            degree_long = match_degree(long, long_time, max_data_index, 0, data_matrix_each_job, pod_num, link_bandwidth)
            degree_short = match_degree(aoi_link, short, short_time, max_data_index, 0, data_matrix_each_job, pod_num, link_bandwidth)
            if degree_long <= degree_short:
                long.append(max_data_index)
            else:
                short.append(max_data_index)
            reverse.remove(max_data_index)
        elif long_time >= train_short and short_time < train_long:
            job_match_degree = np.zeros(len(reverse))
            for i in range(len(reverse)):
                job_match_degree[i] = match_degree(aoi_link, short, short_time, reverse[i], 0, data_matrix_each_job, pod_num, link_bandwidth)
            min_match_index = reverse[np.argmin(job_match_degree)]
            short.append(min_match_index)
            reverse.remove(min_match_index)
        elif long_time < train_short and short_time >= train_long:
            job_match_degree = np.zeros(len(reverse))
            for i in range(len(reverse)):
                job_match_degree[i] = match_degree(aoi_link, long, long_time, reverse[i], 0, data_matrix_each_job, pod_num, link_bandwidth)
            min_match_index = reverse[np.argmin(job_match_degree)]
            long.append(min_match_index)
            reverse.remove(min_match_index)
        elif long_time < train_short and short_time < train_long:
            long_set = np.zeros(len(reverse))
            short_set = np.zeros(len(reverse))
            for i in range(len(reverse)):
                degree_long = match_degree(aoi_link, long, long_time, reverse[i], 0, data_matrix_each_job, pod_num, link_bandwidth)
                degree_short = match_degree(aoi_link, short, short_time, reverse[i], 0, data_matrix_each_job, pod_num, link_bandwidth)
                long_set[i] = degree_long
                short_set[i] = degree_short
            min_long, min_long_index = np.min(long_set), np.argmin(long_set)
            min_short, min_short_index = np.min(short_set), np.argmin(short_set)
            if min_long > min_short:
                remove_index = min_short_index
                short.append(reverse[min_short_index])
            else:
                remove_index = min_long_index
                long.append(reverse[min_long_index])
            reverse.remove(reverse[remove_index])
    long_time = count_group_time(aoi_link, long, data_matrix_each_job, pod_num, 0, link_bandwidth)
    short_time = count_group_time(aoi_link, short, data_matrix_each_job, pod_num, 0, link_bandwidth)
    return long, short, long_time, short_time


def count_time(link_m, data_m, pod_num, bandwidth):
    t_matrix = np.zeros([pod_num, pod_num])
    for u in range(pod_num):
        for v in range(pod_num):
            if data_m[u][v] > 0:
                t_matrix[u][v] = data_m[u][v] / (link_m[u][v] * bandwidth)
    return t_matrix


def choose_link(link_m, data_m_1, data_m_2, pod_num, bandwidth, train_1, train_2, max_t1, max_t2, degree, port):
    t_add_link = -1 * np.ones([pod_num, pod_num])
    for u in range(pod_num):
        for v in range(pod_num):
            if degree[u] == port or degree[v] == port:
                continue
            elif degree[u] < port and degree[v] < port:
                link_add = copy.deepcopy(link_m)
                link_add[u][v] += 1
                t1 = count_time(link_add, data_m_1, pod_num, bandwidth)
                t2 = count_time(link_add, data_m_2, pod_num, bandwidth)
                delta_t = max(np.max(t1), train_2) + max(np.max(t2), train_1) - max(max_t1, train_2) - max(max_t2, train_1)
                t_add_link[u][v] = delta_t
    max_link, max_index = np.max(t_add_link), np.argmax(t_add_link)
    max_row, max_col = int(max_index / pod_num), int(max_index % pod_num)
    if max_link <= 0:
        return -1, -1
    else:
        return max_row, max_col


def topo_each_group(data_matrix_each_job, pod_num, boolean_change, group1, group2, train_1, train_2, local_solution, port, link_bandwidth):
    if boolean_change == 0:
        data_group1 = np.zeros([pod_num, pod_num])
        data_group2 = np.zeros([pod_num, pod_num])
        link_matrix = np.zeros([pod_num, pod_num])
        for i in range(len(local_solution)):
            if i in group1:
                data_group1 += data_matrix_each_job[i]
            if i in group2:
                data_group2 += data_matrix_each_job[i]
        for u in range(pod_num):
            for v in range(pod_num):
                if data_group1[u][v] + data_group2[u][v] > 0:
                    link_matrix[u][v] = 1
        degree_pod = np.zeros(pod_num)
        for u in range(pod_num):
            degree_pod[u] = sum([link_matrix[u][v] + link_matrix[v][u] for v in range(pod_num)])
            if degree_pod[u] > pod_num:
                return -1
        while 1:
            t1_matrix = count_time(link_matrix, data_group1, pod_num, link_bandwidth)
            t2_matrix = count_time(link_matrix, data_group2, pod_num, link_bandwidth)
            max_t1, max_t1_index = np.max(t1_matrix), np.argmax(t1_matrix)
            max_t2, max_t2_index = np.max(t2_matrix), np.argmax(t2_matrix)
            if max_t1 < train_2 and max_t2 < train_1:
                return link_matrix
            elif max_t1 < train_2 and max_t2 > train_1:
                max_t2_row, max_t2_col = int(max_t2_index / pod_num), int(max_t2_index % pod_num)
                if degree_pod[max_t2_row] == port_num or degree_pod[max_t2_col] == port_num:
                    return link_matrix
                elif degree_pod[max_t2_row] < port_num and degree_pod[max_t2_col] < port_num:
                    link_matrix[max_t2_row][max_t2_col] += 1
                degree_pod[max_t2_row] += 1
                degree_pod[max_t2_col] += 1
            elif max_t1 > train_2 and max_t2 < train_1:
                max_t1_row, max_t1_col = int(max_t1_index / pod_num), int(max_t1_index % pod_num)
                if degree_pod[max_t1_row] == port_num or degree_pod[max_t1_col] == port_num:
                    return link_matrix
                elif degree_pod[max_t1_row] < port_num and degree_pod[max_t1_col] < port_num:
                    link_matrix[max_t1_row][max_t1_col] += 1
                degree_pod[max_t1_row] += 1
                degree_pod[max_t1_col] += 1
            elif max_t1 > train_2 and max_t2 > train_1:
                link_row, link_col = (
                    choose_link(link_matrix, data_group1, data_group2, pod_num, link_bandwidth, train_1, train_2, max_t1, max_t2, degree_pod, pod_num))
                if link_row == -1:
                    return link_matrix
                else:
                    link_matrix[link_row][link_col] += 1


def group_algorithm(aoi_link, local_solution, link_bandwidth, t_train, data_per_worker, pod_num, boolean_change, port_num):
    sort_train = np.argsort(t_train)
    # print(non_conflict(local_solution), sort_train, t_train)
    t_test = []
    long_group_set = []
    short_group_set = []
    if boolean_change == 0:
        data_matrix = local_data_revert(local_solution, data_per_worker, 0, pod_num)
        # aoi_link = topo_count_fix(data_matrix)
        print(aoi_link)
        for i in range(1, len(sort_train)):
            long_group = [sort_train[k] for k in range(i, len(sort_train))]
            short_group = [sort_train[i - 1]]
            reverse_job = [sort_train[k] for k in range(0, i - 1)]
            t_long_train = max(t_train)
            t_short_train = t_train[short_group[0]]
            # print(t_train)
            long_group, short_group, long_time, short_time = (
                job_allocate_fix(aoi_link, long_group, short_group, reverse_job, data_matrix, pod_num, link_bandwidth, t_long_train, t_short_train, local_solution, port_num))
            if long_time <= t_short_train and short_time <= t_long_train:
                t_test.append(t_short_train + t_long_train)
                long_group_set.append(long_group)
                short_group_set.append(short_group)
                break
            else:
                t_test.append(max(long_time, t_short_train) + max(short_time, t_long_train))
                long_group_set.append(long_group)
                short_group_set.append(short_group)
        test = np.array([t_test[i] for i in range(len(t_test))])
        min_t_feasible, min_t_feasible_index = np.min(test), np.argmin(test)
        return min_t_feasible, long_group_set[min_t_feasible_index], short_group_set[min_t_feasible_index]


def topo_count_fix(data_matrix_each_job, pod_number, port_num):
    d_sum = sum(np.array([data_matrix_each_job[i] for i in range(len(data_matrix_each_job))]))
    link = np.zeros([pod_number, pod_number])
    degree = np.zeros(pod_number)
    t_he = np.zeros([pod_number, pod_number])
    for u in range(0, pod_number):
        for v in range(0, pod_number):
            if d_sum[u][v] > 0:
                link[u][v] += 1
                t_he[u][v] = d_sum[u][v]
                degree[u] += 1
                degree[v] += 1

    while 1:
        max_index = np.argmax(t_he)
        max_row = int(max_index / pod_number)
        max_col = max_index % pod_number
        if degree[max_row] > port_num or degree[max_col] > port_num:
            print("invalid")
            break
        if t_he[max_row][max_col] == 0:
            break
        if degree[max_row] == port_num:
            for u in range(pod_number):
                t_he[max_row][u] = 0
        if degree[max_col] == port_num:
            for u in range(pod_number):
                t_he[u][max_col] = 0
        if degree[max_row] < port_num and degree[max_col] < port_num:
            link[max_row][max_col] += 1
            degree[max_row] += 1
            degree[max_col] += 1
            t_he[max_row][max_col] = d_sum[max_row][max_col] / link[max_row][max_col]
    return link


# job_number = 30
# job1 = generate_job(job_number)
# # job = info[2]
# # job1 = job[0:30]
# all_job_index = [job1[i][0] for i in range(0, len(job1))]
# single_link_out, sum_traffic_out = traffic_count(job1)
# usage = 0.4
# iter_num = 10
# flop = 275
# train_time = job_set_train(job1, flop, usage)
# pod_number = 8
# b_link = 30
# port_num = 8
# t_recon = 0.1
# solution_out, undeploy_out, fix_job, unfix_job = deploy_server(all_job_index, job1, pod_number, 512, 4)
# # print(job1)
# # print(solution_out, undeploy_out, fix_job, unfix_job, single_link_out)
#
# all_data, link = ILP_new.ilp_new(solution_out, fix_job, unfix_job, train_time, len(job1), pod_number, b_link, t_recon, single_link_out, port_num)
# print(group_algorithm(link, solution_out, b_link, train_time, single_link_out, pod_number, 0))
# LP_relax.lp_relax(solution_out, fix_job, unfix_job, train_time, len(job1), pod_number, b_link, t_recon, single_link_out, port_num,t_feasible)
