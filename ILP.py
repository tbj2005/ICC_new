import itertools
import gurobipy as gp
from gurobipy import *
import numpy as np

m = 0.01
M = 10000


def sgn(num):
    if num >= 1:
        return 1
    else:
        return 0


def sub_group(group):
    sub = []
    for n in range(2, len(group)):
        sub += list(itertools.combinations(group, n))
    return sub


def ilp_model(worker_node, ps_node, pod_num, traffic_single_link, b_tor, b_oxc, rar, ps, ep, per_oxc_port, train_time):
    model = gp.Model("ICC")
    model.update()
    t_round = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t_round")
    t1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t1")
    t2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t2")
    t3 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t3")
    t4 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t4")
    t_comm1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t_comm1")
    t_comm2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t_comm2")
    t_comm3 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t_comm3")
    t_comm4 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t_comm4")
    train1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="train1")
    train2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="train2")
    train3 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="train3")
    train4 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="train4")
    d_inter = model.addVars(4, pod_num, pod_num, vtype=GRB.CONTINUOUS, name="d_inter")
    d_intra = model.addVars(4, pod_num, vtype=GRB.CONTINUOUS, name="d_intra")
    link = model.addVars(4, pod_num, pod_num, vtype=GRB.INTEGER, name="link")
    r = model.addVars(4, len(worker_node), pod_num, pod_num, vtype=GRB.BINARY, name="r")
    a = model.addVars(len(worker_node), vtype=GRB.BINARY, name="a")
    x = model.addVars(len(worker_node), vtype=GRB.BINARY, name="x")

    model.setObjective(t_round, GRB.MINIMIZE)

    model.addConstr(t_round == t1 + t2 + t2 + t4, name="")

    model.addConstr(t1 >= t_comm1, name="")
    model.addConstr(t2 >= t_comm2, name="")
    model.addConstr(t3 >= t_comm3, name="")
    model.addConstr(t4 >= t_comm4, name="")
    model.addConstr(t1 >= train1, name="")
    model.addConstr(t2 >= train2, name="")
    model.addConstr(t3 >= train3, name="")
    model.addConstr(t4 >= train4, name="")

    model.addConstrs(
        (t_comm1 * b_oxc * link[0, i, j] >= d_inter[0, i, j] for i in range(0, pod_num) for j in range(0, pod_num)))
    model.addConstrs((t_comm1 * b_tor >= d_intra[0, i] for i in range(0, pod_num)))
    model.addConstrs(
        (t_comm2 * b_oxc * link[1, i, j] >= d_inter[1, i, j] for i in range(0, pod_num) for j in range(0, pod_num)))
    model.addConstrs((t_comm2 * b_tor >= d_intra[1, i] for i in range(0, pod_num)))
    model.addConstrs(
        (t_comm3 * b_oxc * link[2, i, j] >= d_inter[2, i, j] for i in range(0, pod_num) for j in range(0, pod_num)))
    model.addConstrs((t_comm3 * b_tor >= d_intra[2, i] for i in range(0, pod_num)))
    model.addConstrs(
        (t_comm4 * b_oxc * link[3, i, j] >= d_inter[3, i, j] for i in range(0, pod_num) for j in range(0, pod_num)))
    model.addConstrs((t_comm4 * b_tor >= d_intra[3, i] for i in range(0, pod_num)))

    all_sub = []
    sgn_node = np.zeros([len(worker_node), pod_num])
    node_num = np.zeros(len(worker_node))
    for i in range(0, len(worker_node)):
        non_zero = np.array([k for k in worker_node[i] if k > 0])
        all_sub.append(sub_group(non_zero))
        node_num[i] = sum(worker_node[i])
        for j in range(0, pod_num):
            sgn_node[i][j] = sgn(worker_node[i][j])
        
    for n in range(0, len(worker_node)):
        model.addConstrs(
            (quicksum(r[0, n, i, j] for i in range(0, pod_num)) == x[n] * rar[n] * sgn_node[n][j] for j in
             range(0, pod_num)))
        model.addConstrs(
            (quicksum(r[0, n, j, i] for i in range(0, pod_num)) == x[n] * rar[n] * sgn_node[n][j] for j in
             range(0, pod_num)))
        model.addConstrs(
            (quicksum(r[1, n, i, j] for i in range(0, pod_num)) == (1 - x[n]) * rar[n] * sgn_node[n][j] for j in
             range(0, pod_num)))
        model.addConstrs(
            (quicksum(r[1, n, j, i] for i in range(0, pod_num)) == (1 - x[n]) * rar[n] * sgn_node[n][j] for j in
             range(0, pod_num)))
        model.addConstrs(
            (quicksum(r[2, n, i, j] for i in range(0, pod_num)) == x[n] * rar[n] * sgn_node[n][j] for j in
             range(0, pod_num)))
        model.addConstrs(
            (quicksum(r[2, n, j, i] for i in range(0, pod_num)) == x[n] * rar[n] * sgn_node[n][j] for j in
             range(0, pod_num)))
        model.addConstrs(
            (quicksum(r[3, n, i, j] for i in range(0, pod_num)) == (1 - x[n]) * rar[n] * sgn_node[n][j] for j in
             range(0, pod_num)))
        model.addConstrs(
            (quicksum(r[3, n, j, i] for i in range(0, pod_num)) == (1 - x[n]) * rar[n] * sgn_node[n][j] for j in
             range(0, pod_num)))
        model.addConstr(a[n] >= m * (sum(worker_node[n]) - 2))
        model.addConstr(a[n] <= M * (sum(worker_node[n]) - 2))
        model.addConstrs((r[0, n, i, i] == 0 for i in range(0, pod_num)))
        model.addConstrs((r[1, n, i, i] == 0 for i in range(0, pod_num)))
        model.addConstrs((r[2, n, i, i] == 0 for i in range(0, pod_num)))
        model.addConstrs((r[3, n, i, i] == 0 for i in range(0, pod_num)))
        model.addConstrs((quicksum(r[0, n, i, j] * a[n] * rar[n] for i in ele for j in ele) <= len(ele) for ele in all_sub[n]))
        model.addConstrs((quicksum(r[1, n, i, j] * a[n] * rar[n] for i in ele for j in ele) <= len(ele) for ele in all_sub[n]))
        model.addConstrs((quicksum(r[2, n, i, j] * a[n] * rar[n] for i in ele for j in ele) <= len(ele) for ele in all_sub[n]))
        model.addConstrs((quicksum(r[3, n, i, j] * a[n] * rar[n] for i in ele for j in ele) <= len(ele) for ele in all_sub[n]))

    for i in range(0, pod_num):
        model.addConstr(
            quicksum(2 * rar[n] * x[n] * traffic_single_link[n] * worker_node[n][i] for n in range(0, len(worker_node)))
            + quicksum(ps[n] * x[n] * traffic_single_link[n] * worker_node[n][i] for n in range(0, len(worker_node)))
            + quicksum(ep[n] * x[n] * traffic_single_link[n] * worker_node[n][i] * (sum(worker_node[n]) - 1) for n in
                       range(0, len(worker_node))) == d_intra[0, i])
        model.addConstr(
            quicksum(2 * rar[n] * (1 - x[n]) * traffic_single_link[n] * worker_node[n][i] for n in
                     range(0, len(worker_node)))
            + quicksum(
                ps[n] * (1 - x[n]) * traffic_single_link[n] * worker_node[n][i] for n in range(0, len(worker_node)))
            + quicksum(
                ep[n] * (1 - x[n]) * traffic_single_link[n] * worker_node[n][i] * (sum(worker_node[n]) - 1) for n in
                range(0, len(worker_node))) == d_intra[1, i])
        model.addConstr(d_intra[1, i] == d_intra[3, i])
        model.addConstr(d_intra[0, i] == d_intra[2, i])

    for i in range(0, pod_num):
        model.addConstr(quicksum(link[0, i, j] for i in range(0, pod_num) for j in range(0, pod_num)) <= per_oxc_port)
        model.addConstr(quicksum(link[1, i, j] for i in range(0, pod_num) for j in range(0, pod_num)) <= per_oxc_port)
        model.addConstr(quicksum(link[2, i, j] for i in range(0, pod_num) for j in range(0, pod_num)) <= per_oxc_port)
        model.addConstr(quicksum(link[3, i, j] for i in range(0, pod_num) for j in range(0, pod_num)) <= per_oxc_port)

    for i in range(0, pod_num):
        for j in range(0, pod_num):
            if i != j:
                model.addConstr(
                    quicksum(r[0, n, i, j] * traffic_single_link[n] * rar[n] * x[n] for n in range(0, len(worker_node))) +
                    quicksum(x[n] * ps_node[n][i] * worker_node[n][j] * traffic_single_link[n] for n in
                             range(0, len(worker_node)))
                    + quicksum(x[n] * traffic_single_link[n] * ep[n] * worker_node[n][i] * worker_node[n][j] for n in
                               range(0, len(worker_node))) == d_inter[0, i, j])
                model.addConstr(
                    quicksum(r[0, n, i, j] * traffic_single_link[n] * rar[n] * (1 - x[n]) for n in range(0, len(worker_node))) +
                    quicksum((1 - x[n]) * ps_node[n][j] * worker_node[n][i] * traffic_single_link[n] for n in
                             range(0, len(worker_node)))
                    + quicksum((1 - x[n]) * traffic_single_link[n] * ep[n] * worker_node[n][i] * worker_node[n][j] for n in
                               range(0, len(worker_node))) == d_inter[1, i, j])
                model.addConstr(
                    quicksum(r[0, n, i, j] * traffic_single_link[n] * rar[n] * x[n] for n in range(0, len(worker_node))) +
                    quicksum(x[n] * ps_node[n][j] * worker_node[n][i] * traffic_single_link[n] for n in
                             range(0, len(worker_node)))
                    + quicksum(x[n] * traffic_single_link[n] * ep[n] * worker_node[n][i] * worker_node[n][j] for n in
                               range(0, len(worker_node))) == d_inter[2, i, j])
                model.addConstr(
                    quicksum(
                        r[0, n, i, j] * traffic_single_link[n] * rar[n] * (1 - x[n]) for n in range(0, len(worker_node))) +
                    quicksum((1 - x[n]) * ps_node[n][i] * worker_node[n][j] * traffic_single_link[n] for n in
                             range(0, len(worker_node)))
                    + quicksum(
                        (1 - x[n]) * traffic_single_link[n] * ep[n] * worker_node[n][i] * worker_node[n][j] for n in
                        range(0, len(worker_node))) == d_inter[3, i, j])

    model.addConstrs((train1 >= train_time[n] * (1 - x[n]) for n in range(0, len(worker_node))))
    model.addConstrs((train2 >= train_time[n] * x[n] for n in range(0, len(worker_node))))
    model.addConstrs((train3 >= train_time[n] * (1 - x[n]) * (1 - ps[n]) for n in range(0, len(worker_node))))
    model.addConstrs((train3 >= train_time[n] * x[n] * (1 - ps[n]) for n in range(0, len(worker_node))))

    model.setParam("OutputFlag", 1)
    model.Params.LogToConsole = True
    model.optimize()

    for v in model.getVars():
        (name, data) = (v.varName, v.x)
        if name[:7] == "t_round":
            print(name, data)
        if name[:1] == "x":
            print(name, data)
