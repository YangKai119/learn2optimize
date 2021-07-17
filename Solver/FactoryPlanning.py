

import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

def exp1():
    products = ["Prod1", "Prod2", "Prod3", "Prod4", "Prod5", "Prod6", "Prod7"]
    machines = ["grinder", "vertDrill", "horiDrill", "borer", "planer"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

    profit = {"Prod1":10, "Prod2":6, "Prod3":8, "Prod4":4, "Prod5":11, "Prod6":9, "Prod7":3}

    time_req = {
        "grinder": {    "Prod1": 0.5, "Prod2": 0.7, "Prod5": 0.3,
                        "Prod6": 0.2, "Prod7": 0.5 },
        "vertDrill": {  "Prod1": 0.1, "Prod2": 0.2, "Prod4": 0.3,
                        "Prod6": 0.6 },
        "horiDrill": {  "Prod1": 0.2, "Prod3": 0.8, "Prod7": 0.6 },
        "borer": {      "Prod1": 0.05,"Prod2": 0.03,"Prod4": 0.07,
                        "Prod5": 0.1, "Prod7": 0.08 },
        "planer": {     "Prod3": 0.01,"Prod5": 0.05,"Prod7": 0.05 }
    }


    # number of machines down
    down = {("Jan","grinder"): 1, ("Feb", "horiDrill"): 2, ("Mar", "borer"): 1,
            ("Apr", "vertDrill"): 1, ("May", "grinder"): 1, ("May", "vertDrill"): 1,
            ("Jun", "planer"): 1, ("Jun", "horiDrill"): 1}

    # number of each machine available
    installed = {"grinder":4, "vertDrill":2, "horiDrill":3, "borer":1, "planer":1}

    # market limitation of sells
    max_sales = {
        ("Jan", "Prod1") : 500,
        ("Jan", "Prod2") : 1000,
        ("Jan", "Prod3") : 300,
        ("Jan", "Prod4") : 300,
        ("Jan", "Prod5") : 800,
        ("Jan", "Prod6") : 200,
        ("Jan", "Prod7") : 100,
        ("Feb", "Prod1") : 600,
        ("Feb", "Prod2") : 500,
        ("Feb", "Prod3") : 200,
        ("Feb", "Prod4") : 0,
        ("Feb", "Prod5") : 400,
        ("Feb", "Prod6") : 300,
        ("Feb", "Prod7") : 150,
        ("Mar", "Prod1") : 300,
        ("Mar", "Prod2") : 600,
        ("Mar", "Prod3") : 0,
        ("Mar", "Prod4") : 0,
        ("Mar", "Prod5") : 500,
        ("Mar", "Prod6") : 400,
        ("Mar", "Prod7") : 100,
        ("Apr", "Prod1") : 200,
        ("Apr", "Prod2") : 300,
        ("Apr", "Prod3") : 400,
        ("Apr", "Prod4") : 500,
        ("Apr", "Prod5") : 200,
        ("Apr", "Prod6") : 0,
        ("Apr", "Prod7") : 100,
        ("May", "Prod1") : 0,
        ("May", "Prod2") : 100,
        ("May", "Prod3") : 500,
        ("May", "Prod4") : 100,
        ("May", "Prod5") : 1000,
        ("May", "Prod6") : 300,
        ("May", "Prod7") : 0,
        ("Jun", "Prod1") : 500,
        ("Jun", "Prod2") : 500,
        ("Jun", "Prod3") : 100,
        ("Jun", "Prod4") : 300,
        ("Jun", "Prod5") : 1100,
        ("Jun", "Prod6") : 500,
        ("Jun", "Prod7") : 60,
    }

    holding_cost = 0.5
    max_inventory = 100
    store_target = 50
    hours_per_month = 2*8*24

    factory = gp.Model('Factory Planning I')

    make = factory.addVars(months, products, name="Make") # quantity manufactured
    store = factory.addVars(months, products, ub=max_inventory, name="Store") # quantity stored
    sell = factory.addVars(months, products, ub=max_sales, name="Sell") # quantity sold


    Balance0 = factory.addConstrs((make[months[0], product] == sell[months[0], product]
                      + store[months[0], product] for product in products), name="Initial_Balance")

    Balance = factory.addConstrs((store[months[months.index(month) -1], product] +
                    make[month, product] == sell[month, product] + store[month, product]
                    for product in products for month in months
                    if month != months[0]), name="Balance")

    TargetInv = factory.addConstrs((store[months[-1], product] == store_target for product in products),  name="End_Balance")

    MachineCap = factory.addConstrs((gp.quicksum(time_req[machine][product] * make[month, product]
                                 for product in time_req[machine])
                        <= hours_per_month * (installed[machine] - down.get((month, machine), 0))
                        for machine in machines for month in months),
                       name = "Capacity")


    obj = gp.quicksum(profit[product] * sell[month, product] -  holding_cost * store[month, product]
                   for month in months for product in products)

    factory.setObjective(obj, GRB.MAXIMIZE)

    factory.optimize()

    rows = months.copy()
    columns = products.copy()
    make_plan = pd.DataFrame(columns=columns, index=rows, data=0.0)

    for month, product in make.keys():
        if (abs(make[month, product].x) > 1e-6):
            make_plan.loc[month, product] = np.round(make[month, product].x, 1)
    print(make_plan)

    rows = months.copy()
    columns = products.copy()
    sell_plan = pd.DataFrame(columns=columns, index=rows, data=0.0)

    for month, product in sell.keys():
        if (abs(sell[month, product].x) > 1e-6):
            sell_plan.loc[month, product] = np.round(sell[month, product].x, 1)
    print(sell_plan)


    rows = months.copy()
    columns = products.copy()
    store_plan = pd.DataFrame(columns=columns, index=rows, data=0.0)

    for month, product in store.keys():
        if (abs(store[month, product].x) > 1e-6):
            store_plan.loc[month, product] = np.round(store[month, product].x, 1)
    print(store_plan)

def exp2():
    products = ["Prod1", "Prod2", "Prod3", "Prod4", "Prod5", "Prod6", "Prod7"]
    machines = ["grinder", "vertDrill", "horiDrill", "borer", "planer"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

    profit = {"Prod1": 10, "Prod2": 6, "Prod3": 8, "Prod4": 4, "Prod5": 11, "Prod6": 9, "Prod7": 3}

    time_req = {
        "grinder": {"Prod1": 0.5, "Prod2": 0.7, "Prod5": 0.3,
                    "Prod6": 0.2, "Prod7": 0.5},
        "vertDrill": {"Prod1": 0.1, "Prod2": 0.2, "Prod4": 0.3,
                      "Prod6": 0.6},
        "horiDrill": {"Prod1": 0.2, "Prod3": 0.8, "Prod7": 0.6},
        "borer": {"Prod1": 0.05, "Prod2": 0.03, "Prod4": 0.07,
                  "Prod5": 0.1, "Prod7": 0.08},
        "planer": {"Prod3": 0.01, "Prod5": 0.05, "Prod7": 0.05}
    }

    installed = {"grinder": 4, "vertDrill": 2, "horiDrill": 3, "borer": 1, "planer": 1}

    down_req = {"grinder": 2, "vertDrill": 2, "horiDrill": 3, "borer": 1, "planer": 1}

    max_sales = {
        ("Jan", "Prod1"): 500,
        ("Jan", "Prod2"): 1000,
        ("Jan", "Prod3"): 300,
        ("Jan", "Prod4"): 300,
        ("Jan", "Prod5"): 800,
        ("Jan", "Prod6"): 200,
        ("Jan", "Prod7"): 100,
        ("Feb", "Prod1"): 600,
        ("Feb", "Prod2"): 500,
        ("Feb", "Prod3"): 200,
        ("Feb", "Prod4"): 0,
        ("Feb", "Prod5"): 400,
        ("Feb", "Prod6"): 300,
        ("Feb", "Prod7"): 150,
        ("Mar", "Prod1"): 300,
        ("Mar", "Prod2"): 600,
        ("Mar", "Prod3"): 0,
        ("Mar", "Prod4"): 0,
        ("Mar", "Prod5"): 500,
        ("Mar", "Prod6"): 400,
        ("Mar", "Prod7"): 100,
        ("Apr", "Prod1"): 200,
        ("Apr", "Prod2"): 300,
        ("Apr", "Prod3"): 400,
        ("Apr", "Prod4"): 500,
        ("Apr", "Prod5"): 200,
        ("Apr", "Prod6"): 0,
        ("Apr", "Prod7"): 100,
        ("May", "Prod1"): 0,
        ("May", "Prod2"): 100,
        ("May", "Prod3"): 500,
        ("May", "Prod4"): 100,
        ("May", "Prod5"): 1000,
        ("May", "Prod6"): 300,
        ("May", "Prod7"): 0,
        ("Jun", "Prod1"): 500,
        ("Jun", "Prod2"): 500,
        ("Jun", "Prod3"): 100,
        ("Jun", "Prod4"): 300,
        ("Jun", "Prod5"): 1100,
        ("Jun", "Prod6"): 500,
        ("Jun", "Prod7"): 60,
    }

    holding_cost = 0.5
    max_inventory = 100
    store_target = 50
    hours_per_month = 2 * 8 * 24

    factory = gp.Model('Factory Planning II')

    make = factory.addVars(months, products, name="Make")  # quantity manufactured
    store = factory.addVars(months, products, ub=max_inventory, name="Store")  # quantity stored
    sell = factory.addVars(months, products, ub=max_sales, name="Sell")  # quantity sold
    repair = factory.addVars(months, machines, vtype=GRB.INTEGER, ub=down_req, name="Repair")  # number of machines down

    Balance0 = factory.addConstrs((make[months[0], product] == sell[months[0], product]
                                   + store[months[0], product] for product in products), name="Initial_Balance")

    Balance = factory.addConstrs((store[months[months.index(month) - 1], product] +
                                  make[month, product] == sell[month, product] + store[month, product]
                                  for product in products for month in months
                                  if month != months[0]), name="Balance")

    TargetInv = factory.addConstrs((store[months[-1], product] == store_target for product in products),
                                   name="End_Balance")

    MachineCap = factory.addConstrs((gp.quicksum(time_req[machine][product] * make[month, product]
                                                 for product in time_req[machine])
                                     <= hours_per_month * (installed[machine] - repair[month, machine])
                                     for machine in machines for month in months),
                                    name="Capacity")

    Maintenance = factory.addConstrs((repair.sum('*', machine) == down_req[machine] for machine in machines),
                                     "Maintenance")

    obj = gp.quicksum(profit[product] * sell[month, product] - holding_cost * store[month, product]
                      for month in months for product in products)

    factory.setObjective(obj, GRB.MAXIMIZE)

    factory.optimize()

    rows = months.copy()
    columns = products.copy()
    make_plan = pd.DataFrame(columns=columns, index=rows, data=0.0)

    for month, product in make.keys():
        if (abs(make[month, product].x) > 1e-6):
            make_plan.loc[month, product] = np.round(make[month, product].x, 1)
    print(make_plan)

    rows = months.copy()
    columns = products.copy()
    sell_plan = pd.DataFrame(columns=columns, index=rows, data=0.0)

    for month, product in sell.keys():
        if (abs(sell[month, product].x) > 1e-6):
            sell_plan.loc[month, product] = np.round(sell[month, product].x, 1)
    print(sell_plan)

    rows = months.copy()
    columns = products.copy()
    store_plan = pd.DataFrame(columns=columns, index=rows, data=0.0)

    for month, product in store.keys():
        if (abs(store[month, product].x) > 1e-6):
            store_plan.loc[month, product] = np.round(store[month, product].x, 1)
    print(store_plan)

    rows = months.copy()
    columns = machines.copy()
    repair_plan = pd.DataFrame(columns=columns, index=rows, data=0.0)

    for month, machine in repair.keys():
        if (abs(repair[month, machine].x) > 1e-6):
            repair_plan.loc[month, machine] = repair[month, machine].x
    print(repair_plan)


if __name__ == '__main__':
    exp2()

