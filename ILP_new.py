import gurobipy as gp
from gurobipy import *
import numpy as np
from itertools import permutations
from itertools import product

m = 0.001
M = 100000


def sub_group(group):
    sub = []
    for n in range(2, len(group)):
        sub += list(itertools.combinations(group, n))
    return sub


def generate_all_possible_paths(nodes, source, destination):
    """
    生成从源节点到目的节点的所有可能路径（无环路）

    参数:
    nodes: 所有节点的列表（包括源和目的节点）
    source: 源节点
    destination: 目的节点

    返回:
    所有可能路径的列表，每条路径是一个节点序列
    """
    # 移除源和目的节点，得到中间节点
    other_nodes = [n for n in nodes if n != source and n != destination]

    all_paths = []

    # 考虑不同长度的路径（从0个中间节点到全部中间节点）
    for k in range(2):
        # 生成所有可能的中间节点排列组合
        for intermediates in permutations(other_nodes, k):
            # 构建完整路径：源 + 中间节点 + 目的
            path = [source] + list(intermediates) + [destination]
            all_paths.append(path)

    return all_paths


def path_matrix_count(node):
    path_matrix = np.empty([node, node], dtype=object)
    node_list = [i for i in range(node)]
    for u in range(node):
        for v in range(node):
            if u == v:
                path_matrix[u][v] = []
            else:
                path_matrix[u][v] = generate_all_possible_paths(node_list, u, v)
    return path_matrix


def ilp_new(fj, ufj, t_train, num_job, num_pod, b_link, data_per_worker, port_num, local_solution):
    """
    ILP求解代码
    :param local_solution:
    :param port_num: 每个pod的OXC端口数目
    :param data_per_worker:单 worker 数据量
    :param fj:固定拓扑业务
    :param ufj:非固定拓扑业务，当前只考虑 ring
    :param t_train:业务训练时间
    :param num_job:业务数目
    :param num_pod:pod数目
    :param b_link:单连接带宽
    :return:
    """
    model = gp.Model("ICC_new")
    p_matrix = path_matrix_count(num_pod)
    data_job = []
    z_job = []
    z_1 = [[] for i in range(len(local_solution))]
    abs_delta = [[] for i in range(len(local_solution))]
    z_ele = [[] for i in range(len(local_solution))]
    z_data_job = [[] for i in range(len(local_solution))]
    print(data_per_worker)
    for i in range(len(local_solution)):
        data_matrix_all = []
        if ufj[i] == 1:
            worker = [k for k in range(len(local_solution[i][1])) if local_solution[i][1][k] > 0]
            first_worker = worker[0]
            reserve_worker = [k for k in worker if k != first_worker]
            all_ring = list(permutations(reserve_worker))
            all_ring = [[first_worker] + list(all_ring[k]) for k in range(len(all_ring))]
            for j in range(len(all_ring)):
                worker_sort = list(all_ring[j])
                worker_sort += [worker_sort[0]]
                path_flow = []
                for u in range(len(worker_sort) - 1):
                    path_flow.append(p_matrix[worker_sort[u]][worker_sort[u + 1]])
                for combo in product(*path_flow):
                    data_matrix_single = np.zeros([num_pod, num_pod])
                    for k in range(len(combo)):
                        path = combo[k]
                        for c in range(len(path) - 1):
                            data_matrix_single[path[c]][path[c + 1]] += data_per_worker[i]
                    data_matrix_all.append(data_matrix_single)
                    z_ele[i].append(model.addVars(num_pod, num_pod, vtype=GRB.BINARY, name="z_ele"))
                    z_1[i].append(model.addVars(num_pod, num_pod, vtype=GRB.BINARY, name="z_1"))
                    abs_delta[i].append(model.addVars(num_pod, num_pod, vtype=GRB.CONTINUOUS, name="abs_delta"))
                    z_data_job[i].append(model.addVars(num_pod, num_pod, vtype=GRB.CONTINUOUS, name="z_data"))
        if fj[i] == 1:
            ps = local_solution[i][0]
            worker = [k for k in range(len(local_solution[i][1])) if local_solution[i][1][k] > 0]
            path_flow = []
            for u in worker:
                if ps == u:
                    continue
                else:
                    path_flow.append(p_matrix[ps][u])
                    path_flow.append(p_matrix[u][ps])
            for combo in product(*path_flow):
                data_matrix_single = np.zeros([num_pod, num_pod])
                for k in range(len(combo)):
                    path = combo[k]
                    for c in range(len(path) - 1):
                        data_matrix_single[path[c]][path[c + 1]] += data_per_worker[i]
                data_matrix_all.append(data_matrix_single)
                z_ele[i].append(model.addVars(num_pod, num_pod, vtype=GRB.BINARY, name="z_ele"))
                z_1[i].append(model.addVars(num_pod, num_pod, vtype=GRB.BINARY, name="z_1"))
                abs_delta[i].append(model.addVars(num_pod, num_pod, vtype=GRB.CONTINUOUS, name="abs_delta"))
                z_data_job[i].append(model.addVars(num_pod, num_pod, vtype=GRB.CONTINUOUS, name="z_data"))
        data_job.append(data_matrix_all)
        z_job.append(model.addVars(len(data_matrix_all), vtype=GRB.BINARY, name="z_job"))
        # z_data_job.append(model.addVars(len(data_matrix_all), vtype=GRB.BINARY, name="z_data_job"))
    link_matrix = np.zeros([num_pod, num_pod])
    d_matrix = np.array([np.zeros([num_pod, num_pod]) for _ in range(num_job)])
    model.update()
    t_round = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t_round")
    t1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t1")
    t2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t2")
    t1_comp = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t1_comp")
    t2_comp = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t2_comp")
    # 计算时间
    t1_comm = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t1_comm")
    t2_comm = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t2_comm")

    delta_a_u_v = model.addVars(port_num + 1, num_pod, num_pod, vtype=GRB.BINARY, name="delta_a_u_v")
    w_a_u_v = model.addVars(2, port_num + 1, num_pod, num_pod, lb=0, vtype=GRB.CONTINUOUS, name="w_a_u_v")
    kd_u_v = model.addVars(2, num_job, num_pod, num_pod, lb=0, vtype=GRB.CONTINUOUS, name="kd_u_v")
    # 通信时间，只考虑 AOI 中的时间，下层时间先不管
    link = model.addMVar((num_pod, num_pod), lb=0, vtype=GRB.INTEGER, name="L")
    k1 = model.addVars(num_job, vtype=GRB.BINARY, name="k1")
    # k1=1时，comp一组，comm二组，k2=1时，comp二组，comm一组，先不考虑多跳转发，all-to-all采用环通信以节约端口
    k2 = model.addVars(num_job, vtype=GRB.BINARY, name="k2")
    r = model.addVar(vtype=GRB.BINARY, name="r")
    # r=1时说明会重构，否则不会
    d = model.addMVar((num_job, num_pod, num_pod), vtype=GRB.CONTINUOUS, name="d")
    # delta_link = model.addMVar((num_pod, num_pod), vtype=GRB.INTEGER, name="delta_link")
    b_data = model.addMVar((num_job, num_pod, num_pod), vtype=GRB.BINARY, name="b_data")
    print(t_train)
    model.setObjective(t_round, GRB.MINIMIZE)
    model.addConstr(t_round >= t1_comm + t2_comm, name="")
    model.addConstr(t_round >= t1_comm + t1_comp, name="")
    model.addConstr(t_round >= t2_comp + t2_comm, name="")
    model.addConstr(t_round >= t1_comp + t2_comp, name="")

    # model.addConstr(t1 >= t1_comm, name="")
    # model.addConstr(t1 >= t1_comp, name="")
    # model.addConstr(t2 >= t2_comm, name="")
    # model.addConstr(t2 >= t2_comp, name="")

    model.addConstrs(k1[i] == 1 - k2[i] for i in range(num_job))  # 业务不是在一组就是在二组

    model.addConstrs(t1_comp >= k1[i] * t_train[i] for i in range(num_job))
    model.addConstrs(t2_comp >= k2[i] * t_train[i] for i in range(num_job))  # 每阶段的计算时间大于等于当阶段所有业务的计算时间

    # 在各组组内约束 D <= t * L
    # 第一组

    model.addConstrs(quicksum(delta_a_u_v[a, u, v] for a in range(port_num + 1)) == 1 for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(kd_u_v[0, i, u, v] >= m * k1[i] - m for i in range(num_job) for u in range(num_pod) for v in range(0, num_pod))
    model.addConstrs(kd_u_v[0, i, u, v] <= M * k1[i] for i in range(num_job) for u in range(num_pod) for v in range(0, num_pod))
    for i in range(num_job):
        for u in range(num_pod):
            for v in range(num_pod):
                model.addConstr(d[i, u, v] - M * (1 - k1[i]) <= kd_u_v[0, i, u, v])
                model.addConstr(d[i, u, v] - m * (1 - k1[i]) + m >= kd_u_v[0, i, u, v])
    model.addConstrs(quicksum(a * w_a_u_v[0, a, u, v] for a in range(port_num + 1)) >= quicksum(kd_u_v[0, i, u, v] for i in range(num_job)) for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(M * delta_a_u_v[a, u, v] >= w_a_u_v[0, a, u, v] for a in range(port_num + 1) for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(m * delta_a_u_v[a, u, v] <= w_a_u_v[0, a, u, v] for a in range(port_num + 1) for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(b_link * t1_comm - M * (1 - delta_a_u_v[a, u, v]) <= w_a_u_v[0, a, u, v] for a in range(port_num + 1) for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(b_link * t1_comm - m * (1 - delta_a_u_v[a, u, v]) + m >= w_a_u_v[0, a, u, v] for a in range(port_num + 1) for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(link[u, v] == quicksum(a * delta_a_u_v[a, u, v] for a in range(port_num + 1)) for u in range(num_pod) for v in range(num_pod))

    # 第二组

    model.addConstrs(
        quicksum(delta_a_u_v[a, u, v] for a in range(port_num + 1)) == 1 for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(
        kd_u_v[1, i, u, v] >= m * k2[i] - m for i in range(num_job) for u in range(num_pod) for v in range(0, num_pod))
    model.addConstrs(
        kd_u_v[1, i, u, v] <= M * k2[i] for i in range(num_job) for u in range(num_pod) for v in range(0, num_pod))
    model.addConstrs(
        d[i, u, v] - M * (1 - k2[i]) <= kd_u_v[1, i, u, v] for i in range(num_job) for u in range(num_pod) for v in
        range(num_pod))
    model.addConstrs(
        d[i, u, v] - m * (1 - k2[i]) + m >= kd_u_v[1, i, u, v] for i in range(num_job) for u in range(num_pod) for v in
        range(num_pod))
    model.addConstrs(quicksum(a * w_a_u_v[1, a, u, v] for a in range(port_num + 1)) >= quicksum(
        kd_u_v[1, i, u, v] for i in range(num_job)) for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(
        M * delta_a_u_v[a, u, v] >= w_a_u_v[1, a, u, v] for a in range(port_num + 1) for u in range(num_pod) for v in
        range(num_pod))
    model.addConstrs(
        m * delta_a_u_v[a, u, v] <= w_a_u_v[1, a, u, v] for a in range(port_num + 1) for u in range(num_pod) for v in
        range(num_pod))
    model.addConstrs(
        t2_comm * b_link - M * (1 - delta_a_u_v[a, u, v]) <= w_a_u_v[1, a, u, v] for a in range(port_num + 1) for u in
        range(num_pod) for v in range(num_pod))
    model.addConstrs(
        t2_comm * b_link - m * (1 - delta_a_u_v[a, u, v]) + m >= w_a_u_v[1, a, u, v] for a in range(port_num + 1) for u in
        range(num_pod) for v in range(num_pod))
    model.addConstrs(
        link[u, v] == quicksum(a * delta_a_u_v[a, u, v] for a in range(port_num + 1)) for u in range(num_pod) for v in
        range(num_pod))

    # model.addConstrs(t1_comm * link[u, v] * b_link >= (quicksum(d[i, u, v] * k1[i] for i in range(num_job))) for u in range(num_pod) for v in range(num_pod))
    # model.addConstrs(t2_comm * link[u, v] * b_link >= (quicksum(d[i, u, v] * k2[i] for i in range(num_job))) for u in range(num_pod) for v in range(num_pod))  # 通信时间

    # model.addConstrs(delta_link[u, v] == link[1, u, v] * link[1, u, v] + link[1, u, v] * link[0, u, v] - 2 * link[0, u, v] * link[1, u, v] for u in range(0, num_pod) for v in range(0, num_pod))
    # model.addConstr(r <= quicksum(quicksum(delta_link[u, v] for u in range(num_pod)) for v in range(num_pod)))
    # model.addConstrs(r >= delta_link[u, v] for u in range(num_pod) for v in range(num_pod))
    # 判断是否需要重构

    # for i in range(0, num_job):
    #     if fj[i] != 1:
    #         continue
    #     for x in range(0, num_pod):
    #         if x != local_solution[i][0]:
    #             model.addConstr(d[i, x, local_solution[i][0]] == data_per_worker[i] * local_solution[i][1][x])
    #             model.addConstr(d[i, local_solution[i][0], x] == data_per_worker[i] * local_solution[i][1][x])
    #             # 通过放置约束固定流量矩阵
    #         if x == local_solution[i][0]:
    #             model.addConstr(d[i, x, local_solution[i][0]] == 0)
    #             model.addConstr(d[i, local_solution[i][0], x] == 0)

    for i in range(num_job):
        model.addConstr(quicksum(z_job[i][k] for k in range(len(z_job[i]))) == 1)
        for k in range(len(z_job[i])):
            for u in range(num_pod):
                for v in range(num_pod):
                    model.addConstr(z_data_job[i][k][u, v] <= M * z_1[i][k][u, v])
                    model.addConstr(z_data_job[i][k][u, v] >= m * z_1[i][k][u, v] - m)
                    model.addConstr(z_data_job[i][k][u, v] <= d[i, u, v] - m * (1 - z_1[i][k][u, v]) + m)
                    model.addConstr(z_data_job[i][k][u, v] >= d[i, u, v] - M * (1 - z_1[i][k][u, v]))
                    # model.addConstr(z_1[i][k][u, v] * (d[i, u, v] - data_job[i][k][u, v]) >= 0)
                    model.addConstr(z_data_job[i][k][u, v] >= z_1[i][k][u, v] * data_job[i][k][u, v])
                    model.addConstr(z_1[i][k][u, v] >= m * (d[i, u, v] - data_job[i][k][u, v]))
                    model.addConstr(abs_delta[i][k][u, v] + data_job[i][k][u, v] * (2 * z_1[i][k][u, v] - 1) == 2 * z_data_job[i][k][u, v] - d[i, u, v])
                    model.addConstr(z_ele[i][k][u, v] >= m * abs_delta[i][k][u, v])
                    model.addConstr(z_ele[i][k][u, v] <= M * abs_delta[i][k][u, v])
        model.addConstrs(m * quicksum(quicksum(z_ele[i][k][u, v] for u in range(num_pod)) for v in range(num_pod)) <= 1 - z_job[i][k] for k in range(len(z_job[i])))
        model.addConstrs(
            M * quicksum(quicksum(z_ele[i][k][u, v] for u in range(num_pod)) for v in range(num_pod)) >= 1 -
            z_job[i][k] for k in range(len(z_job[i])))

    # 通过放置约束非固定流量矩阵

    model.addConstrs(quicksum(link[u, v] for u in range(num_pod)) <= port_num for v in range(num_pod))
    model.addConstrs(quicksum(link[u, v] for v in range(num_pod)) <= port_num for u in range(num_pod))
    model.addConstrs(link[u, u] == 0 for u in range(num_pod))
    # 连接约束

    model.setParam("OutputFlag", 1)
    model.Params.LogToConsole = True
    model.optimize()
    # model.computeIIS()
    # model.write("model1.ilp")
    k1 = 0
    k2 = 0
    group = []
    for v in model.getVars():
        (name, data) = (v.varName, v.x)
        if name[:7] == "t_round":
            print(name, data)
        if name[:2] == "k1":
            k_name = name
            k_data = data
            print(k_name, k_data)
            if data == 1:
                group.append(0)
            else:
                group.append(1)
        if name[:7] == "t1_comm" or name[:7] == "t2_comm" or name[:7] == "t1_comp" or name[:7] == "t2_comp":
            print(name, data)
        if name[:2] == "L[":
            link_data = data
            link_name = name
            # print(link_name, link_data)
            link_matrix[int(k1 / num_pod)][k1 % num_pod] = int(link_data)
            k1 += 1
        # if name[:1] == "z":
        #     link_data = data
        #     link_name = name
        #     print(link_name, link_data)
        #     # link_matrix[int(k1 / num_pod)][k1 % num_pod] = int(link_data)
        #     # k1 += 1
        if name[:2] == "d[":
            d_data = data
            d_name = name
            print(d_name, d_data)
            d_matrix[int(k2 / (num_pod * num_pod))][int((k2 % (num_pod * num_pod)) / num_pod)][k2 % num_pod] = d_data
            k2 += 1
    print("link", link_matrix)
    all_data_1 = np.zeros([num_pod, num_pod])
    all_data_2 = np.zeros([num_pod, num_pod])
    for i in range(0, len(d_matrix)):
        all_data_1 += d_matrix[i] * (1 - group[i])
        all_data_2 += d_matrix[i] * group[i]
    print(all_data_1, all_data_2)

    print(all_data_1 + all_data_2)
    return all_data_1 + all_data_2, link_matrix
