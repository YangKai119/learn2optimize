

import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB


def exp1():
    supply = dict({'Liverpool': 150000,
                   'Brighton': 200000})

    through = dict({'Newcastle': 70000,
                    'Birmingham': 50000,
                    'London': 100000,
                    'Exeter': 40000})

    demand = dict({'C1': 50000,
                   'C2': 10000,
                   'C3': 40000,
                   'C4': 35000,
                   'C5': 60000,
                   'C6': 20000})

    arcs, cost = gp.multidict({
        ('Liverpool', 'Newcastle'): 0.5,
        ('Liverpool', 'Birmingham'): 0.5,
        ('Liverpool', 'London'): 1.0,
        ('Liverpool', 'Exeter'): 0.2,
        ('Liverpool', 'C1'): 1.0,
        ('Liverpool', 'C3'): 1.5,
        ('Liverpool', 'C4'): 2.0,
        ('Liverpool', 'C6'): 1.0,
        ('Brighton', 'Birmingham'): 0.3,
        ('Brighton', 'London'): 0.5,
        ('Brighton', 'Exeter'): 0.2,
        ('Brighton', 'C1'): 2.0,
        ('Newcastle', 'C2'): 1.5,
        ('Newcastle', 'C3'): 0.5,
        ('Newcastle', 'C5'): 1.5,
        ('Newcastle', 'C6'): 1.0,
        ('Birmingham', 'C1'): 1.0,
        ('Birmingham', 'C2'): 0.5,
        ('Birmingham', 'C3'): 0.5,
        ('Birmingham', 'C4'): 1.0,
        ('Birmingham', 'C5'): 0.5,
        ('London', 'C2'): 1.5,
        ('London', 'C3'): 2.0,
        ('London', 'C5'): 0.5,
        ('London', 'C6'): 1.5,
        ('Exeter', 'C3'): 0.2,
        ('Exeter', 'C4'): 1.5,
        ('Exeter', 'C5'): 0.5,
        ('Exeter', 'C6'): 1.5
    })



    model = gp.Model('SupplyNetworkDesign')
    flow = model.addVars(arcs, obj=cost, name="flow")
    factories = supply.keys()
    factory_flow = model.addConstrs((gp.quicksum(flow.select(factory, '*')) <= supply[factory]
                                     for factory in factories), name="factory")
    customers = demand.keys()
    customer_flow = model.addConstrs((gp.quicksum(flow.select('*', customer)) == demand[customer]
                                      for customer in customers), name="customer")
    depots = through.keys()
    depot_flow = model.addConstrs((gp.quicksum(flow.select(depot, '*')) == gp.quicksum(flow.select('*', depot))
                                   for depot in depots), name="depot")
    depot_capacity = model.addConstrs((gp.quicksum(flow.select('*', depot)) <= through[depot]
                                       for depot in depots), name="depot_capacity")
    model.optimize()


    product_flow = pd.DataFrame(columns=["From", "To", "Flow"])
    for arc in arcs:
        if flow[arc].x > 1e-6:
            product_flow = product_flow.append({"From": arc[0], "To": arc[1], "Flow": flow[arc].x}, ignore_index=True)
    product_flow.index=[''] * len(product_flow)
    print(product_flow)

def exp2():
    supply = dict({'Liverpool': 150000,
                   'Brighton': 200000})

    through = dict({'Newcastle': 70000,
                    'Birmingham': 50000,
                    'London': 100000,
                    'Exeter': 40000,
                    'Bristol': 30000,
                    'Northampton': 25000})

    opencost = dict({'Newcastle': 10000,
                     'Birmingham': 0,
                     'London': 0,
                     'Exeter': 5000,
                     'Bristol': 12000,
                     'Northampton': 4000})

    demand = dict({'C1': 50000,
                   'C2': 10000,
                   'C3': 40000,
                   'C4': 35000,
                   'C5': 60000,
                   'C6': 20000})

    arcs, cost = gp.multidict({
        ('Liverpool', 'Newcastle'): 0.5,
        ('Liverpool', 'Birmingham'): 0.5,
        ('Liverpool', 'London'): 1.0,
        ('Liverpool', 'Exeter'): 0.2,
        ('Liverpool', 'Bristol'): 0.6,
        ('Liverpool', 'Northampton'): 0.4,
        ('Liverpool', 'C1'): 1.0,
        ('Liverpool', 'C3'): 1.5,
        ('Liverpool', 'C4'): 2.0,
        ('Liverpool', 'C6'): 1.0,
        ('Brighton', 'Birmingham'): 0.3,
        ('Brighton', 'London'): 0.5,
        ('Brighton', 'Exeter'): 0.2,
        ('Brighton', 'Bristol'): 0.4,
        ('Brighton', 'Northampton'): 0.3,
        ('Brighton', 'C1'): 2.0,
        ('Newcastle', 'C2'): 1.5,
        ('Newcastle', 'C3'): 0.5,
        ('Newcastle', 'C5'): 1.5,
        ('Newcastle', 'C6'): 1.0,
        ('Birmingham', 'C1'): 1.0,
        ('Birmingham', 'C2'): 0.5,
        ('Birmingham', 'C3'): 0.5,
        ('Birmingham', 'C4'): 1.0,
        ('Birmingham', 'C5'): 0.5,
        ('London', 'C2'): 1.5,
        ('London', 'C3'): 2.0,
        ('London', 'C5'): 0.5,
        ('London', 'C6'): 1.5,
        ('Exeter', 'C3'): 0.2,
        ('Exeter', 'C4'): 1.5,
        ('Exeter', 'C5'): 0.5,
        ('Exeter', 'C6'): 1.5,
        ('Bristol', 'C1'): 1.2,
        ('Bristol', 'C2'): 0.6,
        ('Bristol', 'C3'): 0.5,
        ('Bristol', 'C5'): 0.3,
        ('Bristol', 'C6'): 0.8,
        ('Northampton', 'C2'): 0.4,
        ('Northampton', 'C4'): 0.5,
        ('Northampton', 'C5'): 0.6,
        ('Northampton', 'C6'): 0.9
    })

    model = gp.Model('SupplyNetworkDesign2')

    depots = through.keys()
    flow = model.addVars(arcs, obj=cost, name="flow")
    open = model.addVars(depots, obj=opencost, vtype=GRB.BINARY, name="open")
    expand = model.addVar(obj=3000, vtype=GRB.BINARY, name="expand")

    open['Birmingham'].lb = 1
    open['London'].lb = 1
    model.objcon = -(opencost['Newcastle'] + opencost['Exeter'])  # Phrased as 'savings from closing'

    factories = supply.keys()
    factory_flow = model.addConstrs((gp.quicksum(flow.select(factory, '*')) <= supply[factory]
                                     for factory in factories), name="factory")
    customers = demand.keys()
    customer_flow = model.addConstrs((gp.quicksum(flow.select('*', customer)) == demand[customer]
                                      for customer in customers), name="customer")
    depot_flow = model.addConstrs((gp.quicksum(flow.select(depot, '*')) == gp.quicksum(flow.select('*', depot))
                                   for depot in depots), name="depot")
    all_but_birmingham = list(set(depots) - set(['Birmingham']))
    depot_capacity = model.addConstrs((gp.quicksum(flow.select(depot, '*')) <= through[depot] * open[depot]
                                       for depot in all_but_birmingham), name="depot_capacity")
    birmingham_capacity = model.addConstr(gp.quicksum(flow.select('*', 'Birmingham')) <= through['Birmingham'] +
                                          20000 * expand, name="birmingham_capacity")

    depot_count = model.addConstr(open.sum() <= 4)

    model.optimize()

    print('List of open depots:', [d for d in depots if open[d].x > 0.5])
    if expand.x > 0.5:
        print('Expand Birmingham')

    product_flow = pd.DataFrame(columns=["From", "To", "Flow"])
    for arc in arcs:
        if flow[arc].x > 1e-6:
            product_flow = product_flow.append({"From": arc[0], "To": arc[1], "Flow": flow[arc].x}, ignore_index=True)
    product_flow.index = [''] * len(product_flow)
    print(product_flow)

if __name__ == "__main__":
    exp2()
