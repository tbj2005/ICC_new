import gurobipy as gp
from gurobipy import *
import numpy as np

m = 0.001
M = 100000


def sub_group(group):
    sub = []
    for n in range(2, len(group)):
        sub += list(itertools.combinations(group, n))
    return sub


def lp_relax(local_solution, fj, ufj, t_train, num_job, num_pod, b_link, t_recon, data_per_worker, port_num, t_feasible):
    """
    ILP求解代码
    :param t_feasible: 可行解的总时间
    :param port_num: 每个pod的OXC端口数目
    :param data_per_worker:单 worker 数据量
    :param local_solution:一个集合，内部的元素是业务的 worker 位置，各元素的结构为二元组，第一个元素为ps位置，若不用ps则为0，第二个元素为数组，存放各pod中该业务的worker数目
    :param fj:固定拓扑业务
    :param ufj:非固定拓扑业务，当前只考虑 ring
    :param t_train:业务训练时间
    :param num_job:业务数目
    :param num_pod:pod数目
    :param b_link:单连接带宽
    :param t_recon:重构时间
    :return:
    """
    link_matrix = np.zeros([num_pod, num_pod])
    d_matrix = np.array([np.zeros([num_pod, num_pod]) for _ in range(num_job)])
    model = gp.Model("ICC_new")
    model.update()
    t_round = model.addVar(lb=0, ub=t_feasible, vtype=GRB.CONTINUOUS, name="t_round")
    t1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t1")
    t2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t2")
    t1_comp = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t1_comp")
    t2_comp = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t2_comp")
    # 计算时间
    t1_comm = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t1_comm")
    t2_comm = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t2_comm")

    delta_a_u_v = model.addVars(2, port_num + 1, num_pod, num_pod, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="delta_a_u_v")
    w_a_u_v = model.addVars(2, port_num + 1, num_pod, num_pod, lb=0, vtype=GRB.CONTINUOUS, name="w_a_u_v")
    kd_u_v = model.addVars(2, num_job, num_pod, num_pod, lb=0, vtype=GRB.CONTINUOUS, name="kd_u_v")
    # 通信时间，只考虑 AOI 中的时间，下层时间先不管
    link = model.addMVar((num_pod, num_pod), lb=0, ub=port_num, vtype=GRB.CONTINUOUS, name="L")
    k1 = model.addVars(num_job, vtype=GRB.CONTINUOUS, name="k1")
    # k1=1时，comp一组，comm二组，k2=1时，comp二组，comm一组，先不考虑多跳转发，all-to-all采用环通信以节约端口
    k2 = model.addVars(num_job, vtype=GRB.CONTINUOUS, name="k2")
    t_link = model.addVars(2, num_pod, num_pod, vtype=GRB.CONTINUOUS, name="t_link")
    # r = model.addVar(vtype=GRB.BINARY, name="r")
    # r=1时说明会重构，否则不会
    d = model.addMVar((num_job, num_pod, num_pod), vtype=GRB.CONTINUOUS, name="d")
    # delta_link = model.addMVar((num_pod, num_pod), vtype=GRB.INTEGER, name="delta_link")
    # b_data = model.addMVar((num_job, num_pod, num_pod), vtype=GRB.BINARY, name="b_data")
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

    model.addConstrs(t1_comm >= t_link[0, u, v] for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(t2_comm >= t_link[1, u, v] for u in range(num_pod) for v in range(num_pod))

    model.addConstrs(quicksum(delta_a_u_v[0, a, u, v] for a in range(port_num + 1)) == 1 for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(kd_u_v[0, i, u, v] >= m * k1[i] - m for i in range(num_job) for u in range(num_pod) for v in range(0, num_pod))
    model.addConstrs(kd_u_v[0, i, u, v] <= M * k1[i] for i in range(num_job) for u in range(num_pod) for v in range(0, num_pod))
    model.addConstrs(d[i, u, v] - M * (1 - k1[i]) <= kd_u_v[0, i, u, v] for i in range(num_job) for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(d[i, u, v] - m * (1 - k1[i]) + m >= kd_u_v[0, i, u, v] for i in range(num_job) for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(quicksum(a * w_a_u_v[0, a, u, v] for a in range(port_num + 1)) >= quicksum(kd_u_v[0, i, u, v] for i in range(num_job)) for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(M * delta_a_u_v[0, a, u, v] >= w_a_u_v[0, a, u, v] for a in range(port_num + 1) for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(m * delta_a_u_v[0, a, u, v] <= w_a_u_v[0, a, u, v] for a in range(port_num + 1) for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(b_link * t_link[0, u, v] - M * (1 - delta_a_u_v[0, a, u, v])  <= w_a_u_v[0, a, u, v] for a in range(port_num + 1) for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(b_link * t_link[0, u, v] - m * (1 - delta_a_u_v[0, a, u, v]) >= w_a_u_v[0, a, u, v] for a in range(port_num + 1) for u in range(num_pod) for v in range(num_pod))
    model.addConstrs(link[u, v] == quicksum(a * delta_a_u_v[0, a, u, v] for a in range(port_num + 1)) for u in range(num_pod) for v in range(num_pod))

    model.addConstrs(
        quicksum(delta_a_u_v[1, a, u, v] for a in range(port_num + 1)) == 1 for u in range(num_pod) for v in range(num_pod))
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
        M * delta_a_u_v[1, a, u, v] >= w_a_u_v[1, a, u, v] for a in range(port_num + 1) for u in range(num_pod) for v in
        range(num_pod))
    model.addConstrs(
        m * delta_a_u_v[1, a, u, v] <= w_a_u_v[1, a, u, v] for a in range(port_num + 1) for u in range(num_pod) for v in
        range(num_pod))
    model.addConstrs(
        t_link[1, u, v] * b_link - M * (1 - delta_a_u_v[1, a, u, v]) <= w_a_u_v[1, a, u, v] for a in range(port_num + 1) for u in
        range(num_pod) for v in range(num_pod))
    model.addConstrs(
        t_link[1, u, v] * b_link - m * (1 - delta_a_u_v[1, a, u, v]) >= w_a_u_v[1, a, u, v] for a in range(port_num + 1) for u in
        range(num_pod) for v in range(num_pod))
    model.addConstrs(
        link[u, v] == quicksum(a * delta_a_u_v[1, a, u, v] for a in range(port_num + 1)) for u in range(num_pod) for v in
        range(num_pod))

    # model.addConstrs(t1_comm * link[u, v] * b_link >= (quicksum(d[i, u, v] * k1[i] for i in range(num_job))) for u in range(num_pod) for v in range(num_pod))
    # model.addConstrs(t2_comm * link[u, v] * b_link >= (quicksum(d[i, u, v] * k2[i] for i in range(num_job))) for u in range(num_pod) for v in range(num_pod))  # 通信时间

    # model.addConstrs(delta_link[u, v] == link[1, u, v] * link[1, u, v] + link[1, u, v] * link[0, u, v] - 2 * link[0, u, v] * link[1, u, v] for u in range(0, num_pod) for v in range(0, num_pod))
    # model.addConstr(r <= quicksum(quicksum(delta_link[u, v] for u in range(num_pod)) for v in range(num_pod)))
    # model.addConstrs(r >= delta_link[u, v] for u in range(num_pod) for v in range(num_pod))
    # 判断是否需要重构

    for i in range(0, num_job):
        if fj[i] != 1:
            continue
        for x in range(0, num_pod):
            if x != local_solution[i][0]:
                model.addConstr(d[i, x, local_solution[i][0]] == data_per_worker[i] * local_solution[i][1][x])
                model.addConstr(d[i, local_solution[i][0], x] == data_per_worker[i] * local_solution[i][1][x])
                # 通过放置约束固定流量矩阵
            if x == local_solution[i][0]:
                model.addConstr(d[i, x, local_solution[i][0]] == 0)
                model.addConstr(d[i, local_solution[i][0], x] == 0)

    for i in range(num_job):
        if ufj[i] == 1:
            worker = [x for x in range(len(local_solution[i][1])) if local_solution[i][1][x] > 0]
            for u in range(0, len(worker)):
                if u == len(worker) - 1:
                    model.addConstr(d[i, worker[u], worker[0]] == data_per_worker[i])
                else:
                    model.addConstr(d[i, worker[u], worker[u] + 1] == data_per_worker[i])
            """
            sub_worker = sub_group(worker)
            print(i, sub_worker)
            model.addConstrs(quicksum(b_data[i, u, v] for v in worker) == 1 for u in worker)
            model.addConstrs(quicksum(b_data[i, u, v] for u in worker) == 1 for v in worker)
            model.addConstrs(b_data[i, u, u] == 0 for u in range(num_pod))
            for j in range(len(sub_worker)):
                model.addConstr(quicksum(b_data[i, u, v] for v in sub_worker[j] for u in sub_worker[j]) <= len(sub_worker[j]) - 1)
            model.addConstrs(d[i, u, v] == b_data[i, u, v] * data_per_worker[i] for u in range(num_pod) for v in range(num_pod))
            """
    # 通过放置约束非固定流量矩阵

    model.addConstrs(quicksum(link[u, v] + link[v, u] for u in range(num_pod)) <= port_num for v in range(num_pod))
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
        if name[:1] == "k":
            k_name = name
            k_data = data
            print(k_name, k_data)
        if name[:7] == "t1_comm" or name[:7] == "t2_comm" or name[:7] == "t1_comp" or name[:7] == "t2_comp":
            print(name, data)
        if name[:2] == "L[":
            link_data = data
            link_name = name
            link_matrix[int(k1 / num_pod)][k1 % num_pod] = link_data
            print(link_name, link_data)
        if name[:2] == "d[":
            d_data = data
            d_name = name
            print(d_name, d_data)
        if name[:11] == "delta_a_u_v":
            print(name, data)
    print("link", link_matrix)
