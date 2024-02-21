import pandas as pd
from scipy.optimize import minimize
from gurobipy import Model, GRB
import numpy as np

# Load the dataset from a CSV file
df = pd.read_csv(r'C:\Users\rahul\Desktop\MBAN\Models\price_response.csv')

# Filter the DataFrame to get the coefficients for Line 1 Product 1 and Line 1 Product 2
line_1_product_1_row = df[(df['Product'] == 'Line 1 Product 1')]
line_1_product_2_row = df[(df['Product'] == 'Line 1 Product 2')]
a1 = line_1_product_1_row['Intercept'].values[0]
b1 = line_1_product_2_row['Intercept'].values[0] 

# Define the objective function without regularization
def objective_function(prices, a1, b1):
    p1, p2 = prices
    return p1 * (a1 - b1 * p1) + p2 * (a1 - b1 * p2)

# Define the constraint function
def constraint_function(prices):
    p1, p2 = prices
    # Constraint: p2 - p1 >= 0
    return p2 - p1

# Define the projection step using Gurobi quadratic programming
def projection_step(prices):
    # Initialize Gurobi model
    m = Model("projection_step")
    
    # Define variables
    p1 = m.addVar(lb=0, name="p1")
    p2 = m.addVar(lb=0, name="p2")
    
    # Define objective function (minimize 0)
    m.setObjective(0, GRB.MINIMIZE)
    
    # Define constraint: p2 - p1 >= 0
    m.addConstr(p2 - p1, GRB.GREATER_EQUAL, 0)

    # Set initial values for variables
    p1.start = prices[0]
    p2.start = prices[1]
    
    # Solve the model
    m.optimize()
    
    # Extract optimal prices
    optimal_prices = [p1.x, p2.x]
    
    return optimal_prices


# Set initial guess for prices (non-negative)
initial_prices = [169, 269]  # Adjust based on your scenario

# Define the constraint
constraint = {'type': 'ineq', 'fun': constraint_function}

# Define options with increased maximum number of iterations
options = {'maxiter': 1000}

# Minimize the objective function with bounds and constraints
result = minimize(objective_function, initial_prices, args=(a1, b1), method='COBYLA', constraints=constraint, options=options)

# Extract optimal prices
optimal_prices = result.x

print("*********PART A**********")
print("Optimal prices (p1, p2):", optimal_prices)

# Set initial guess for prices
b_initial_prices = np.zeros(2)

# Set step size for gradient descent
step_size = 0.001

# Set stopping criterion
stopping_criterion = 1e-6

# Perform projected gradient descent
while True:
    
    # Compute gradient of the objective function
    gradient = np.array([-a1 + 2 * b1 * b_initial_prices[0], -a1 + 2 * b1 * b_initial_prices[1]])
    
    # Update prices using gradient descent step
    next_prices = b_initial_prices - step_size * gradient
    
    # Project prices to satisfy the constraint
    next_prices = projection_step(next_prices)
    
    # Check stopping criterion
    if np.linalg.norm(next_prices - b_initial_prices) < stopping_criterion:
        break
    
    b_initial_prices = next_prices
print("*********PART B**********")

print("Optimal prices (p1, p2):", next_prices)

#*************************************PART C*************************************
# Coefficients for each product line and version
coefficients = {
    "ProductLine1": {
        "Basic": {"a": 1000, "b": 0.1},
        "Advanced": {"a": 1500, "b": 0.15},
        "Premium": {"a": 2000, "b": 0.2}
    },
    "ProductLine2": {
        "Basic": {"a": 1200, "b": 0.12},
        "Advanced": {"a": 1700, "b": 0.17},
        "Premium": {"a": 2200, "b": 0.22}
    },
    "ProductLine3": {
        "Basic": {"a": 1100, "b": 0.11},
        "Advanced": {"a": 1600, "b": 0.16},
        "Premium": {"a": 2100, "b": 0.21}
    }
}
# Initialize Gurobi model
model = Model("QuadraticOptimization")

# Decision variables
prices = {}
for product_line, versions in coefficients.items():
    for version, coeffs in versions.items():
        prices[product_line, version] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"price_{product_line}_{version}")

# Auxiliary binary variables to enforce increasing order constraint
is_increasing = {}
for product_line in coefficients.keys():
    is_increasing[product_line] = model.addVars(["Basic_Advanced", "Advanced_Premium"], vtype=GRB.BINARY)

# Objective function: maximize revenue
model.setObjective(sum(prices[product_line, version] * (coeffs["a"] - coeffs["b"] * prices[product_line, version])
                      for product_line, versions in coefficients.items() for version, coeffs in versions.items()),
                   sense=GRB.MAXIMIZE)

# Constraints: Prices of Basic, Advanced, and Premium versions within each product line must be increasing
for product_line in coefficients.keys():
    model.addConstr(prices[product_line, "Basic"] <= prices[product_line, "Advanced"])
    model.addConstr(prices[product_line, "Advanced"] <= prices[product_line, "Premium"])
    
    # Auxiliary constraints to enforce increasing order
    model.addConstr(prices[product_line, "Basic"] + (1 - is_increasing[product_line]["Basic_Advanced"]) * 10000 >= prices[product_line, "Advanced"])
    model.addConstr(prices[product_line, "Advanced"] + (1 - is_increasing[product_line]["Advanced_Premium"]) * 10000 >= prices[product_line, "Premium"])

# Optimize the model
model.optimize()

# Display optimal prices
print("\n*********PART C**********")
print("Optimal Prices:")
for product_line, versions in coefficients.items():
    for version in versions.keys():
        print(f"{product_line} - {version}: {prices[product_line, version].X}")

# Display optimal revenue
print(f"\nOptimal Revenue: {model.objVal}")


#*************************************PART D*************************************

# Create a Gurobi model
model = Model("Price Optimization")

# Decision variables: price for each product in each line
prices = {}
for product, intercept, sensitivity, _ in df.itertuples(index=False):
    prices[product] = model.addVar(lb=0.0, name=f"price_{product}")

# Objective: maximize total revenue
total_revenue = 0
for product, intercept, sensitivity, _ in df.itertuples(index=False):
    total_revenue += prices[product] * (intercept + sensitivity * prices[product])
model.setObjective(total_revenue, GRB.MAXIMIZE)

# Constraints: Prices within each line must be increasing
for product, group in df.groupby('Product'):
    products_in_line = group['Product'].tolist()
    for i in range(len(products_in_line)-1):
        model.addConstr(prices[products_in_line[i+1]] >= prices[products_in_line[i]])

# Optimize the model
model.optimize()

# Display results
if model.status == GRB.OPTIMAL:
    print("***********PART D***********")
    print(f"Optimal objective {model.objVal}")
    for var in model.getVars():
        print(f"{var.varName}: {var.x}")
    print("Optimal Revenue:", model.objVal)
else:
    
    print("***********PART D***********")
    print("No solution found.")
