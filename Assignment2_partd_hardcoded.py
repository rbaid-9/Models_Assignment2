from gurobipy import Model, GRB

# Data: Product, Intercept, Sensitivity, Capacity
data = {
    ("Line 1", "Product 1"): (35234.54579, -45.89644971, 80020),
    ("Line 1", "Product 2"): (37790.24083, -8.227794173, 89666),
    ("Line 1", "Product 3"): (35675.33322, -7.58443641, 80638),
    ("Line 2", "Product 1"): (37041.38038, -9.033166404, 86740),
    ("Line 2", "Product 2"): (36846.14039, -4.427869206, 84050),
    ("Line 2", "Product 3"): (35827.02375, -2.629060015, 86565),
    ("Line 3", "Product 1"): (39414.26632, -2.421483918, 87051),
    ("Line 3", "Product 2"): (35991.95146, -4.000512401, 85156),
    ("Line 3", "Product 3"): (39313.31703, -2.296622373, 87588),
}

# Create a Gurobi model
model = Model("Price Optimization")

# Decision variables: price for each product in each line
prices = {}
for line, product in data.keys():
    prices[line, product] = model.addVar(lb=0.0, name=f"price_{line}_{product}")

# Objective: maximize total revenue
total_revenue = 0
for line, product in data.keys():
    intercept, sensitivity, capacity = data[(line, product)]
    total_revenue += prices[line, product] * (intercept + sensitivity * prices[line, product])
model.setObjective(total_revenue, GRB.MAXIMIZE)

# Constraints: Prices within each line must be increasing
for line in set(line for line, _ in data.keys()):
    products_in_line = [product for l, product in data.keys() if l == line]
    for i in range(len(products_in_line)-1):
        model.addConstr(prices[line, products_in_line[i+1]] >= prices[line, products_in_line[i]])

# Optimize the model
model.optimize()

# Display results
if model.status == GRB.OPTIMAL:
    for var in model.getVars():
        print(f"{var.varName}: {var.x}")
    print("Optimal Revenue:", model.objVal)
else:
    print("No solution found.")
