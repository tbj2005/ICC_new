import gurobipy as gp
import numpy as np
from gurobipy import *


def lp_stuff(matrix, size):
    model = gp.Model("matrix stuff")
    model.update()
    stuff_matrix = model.addMVar((size, size), vtype=GRB.CONTINUOUS, name="stuff_matrix")
    row_stuff = model.addVars(size, vtype=GRB.CONTINUOUS, name="row_add")
    col_stuff = model.addVars(size, vtype=GRB.CONTINUOUS, name="col_add")

    model.addConstrs(stuff_matrix[i][j] >= matrix[i][j] for i in range(size) for j in range(size))

    model.addConstrs(row_stuff[i] == quicksum(stuff_matrix[i][j] for j in range(size)) for i in range(size))
    model.addConstrs(col_stuff[i] == quicksum(stuff_matrix[j][i] for j in range(size)) for i in range(size))
    for i in range(size):
        model.addConstrs(row_stuff[i] == row_stuff[j] for j in range(size))
        model.addConstrs(col_stuff[i] == col_stuff[j] for j in range(size))
    model.addConstr(row_stuff[0] == col_stuff[0], name="")
    model.addConstrs(stuff_matrix[i][i] == 0 for i in range(size))

    model.Params.LogToConsole = 0
    model.setObjective(row_stuff[0], GRB.MINIMIZE)

    model.optimize()

    output_matrix = np.zeros([size, size])
    k = -1

    for v in model.getVars():
        (name, data) = (v.varName, v.x)
        if name[:12] == "stuff_matrix":
            k += 1
            row = int(k / size)
            col = int(k % size)
            v_name, ele = name, data
            output_matrix[row][col] = ele

    return output_matrix
