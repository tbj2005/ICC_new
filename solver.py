import numpy as np
import gurobipy as gb
from gurobipy import *

model = gb.Model("ICC_model")
model.update()

x1 = model.addVar(vtype=GRB.CONTINUOUS,name="lambda1")
x2 = model.addVar(vtype=GRB.CONTINUOUS,name="lambda1")
x3 = model.addVar(vtype=GRB.CONTINUOUS,name="lambda1")
x4 = model.addVar(vtype=GRB.CONTINUOUS,name="lambda1")
x5 = model.addVar(vtype=GRB.CONTINUOUS,name="lambda1")

model.addConstr(x1 == 1.7 + 0.5 * x2 + 0.35 * x3 + 0.2 * x4 + 0.2 * x5)
model.addConstr(x2 == 2.25 + 0.15 *x1 + 0.1 * x3 + 0.25 * x4 + 0.2 * x5)
model.addConstr(x3 == 3.15 + 0.3 * x1 + 0.1 * x4 + 0.1 * x5)
model.addConstr(x4 == 4 + 0.2 * x1 + 0.4 * x2 + 0.05 * x3 + 0.15 * x5)
model.addConstr(x5 == 1.8 + 0.1 * x1 + 0.05 * x2 + 0.1 * x3 + 0.15 * x4)

model.setObjective(x1, GRB.MINIMIZE)

model.optimize()
i = 0
x = np.zeros(5)
for v in model.getVars():
    (name, data) = (v.varName, v.x)
    if name[:6] == "lambda":
        x[i] = data
        i += 1

print(x)
print(x[0] - (1.7 + 0.5 * x[1] + 0.35 * x[2] + 0.2 * x[3] + 0.2 * x[4]))
print(x[1] - (2.25 + 0.15 *x[0] + 0.1 * x[2] + 0.25 * x[3] + 0.2 * x[4]))
print(x[2] - (3.15 + 0.3 * x[0] + 0.1 * x[3] + 0.1 * x[4]))
print(x[3] - (4 + 0.2 * x[0] + 0.4 * x[1] + 0.05 * x[2] + 0.15 * x[4]))
print(x[4] - (1.8 + 0.1 * x[0] + 0.05 * x[1] + 0.1 * x[2] + 0.15 * x[3]))
