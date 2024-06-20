
m ortools.sat.python import cp_model
import numpy as np


GPUM = 64
CPUM = 96
SSDM = 50

def mp_general(tasks, weights):
    model = cp_model.CpModel()
    num_tasks = len(tasks)
    horizon = sum([task[1][1] for task in tasks]) + (sum([weight[1] for weight in weights]) + sum([weight[3] for weight in weights])) * int(sum([weight[0] for weight in weights]) / GPUM + 1)# + max([max(weight[1:]) for weight in weights])
    makespan = model.NewIntVar(0, horizon, 'makespan')

    # Memory System
    GM = [[model.NewIntVar(0, 1, f'GM_{i}_{t}') for t in range(horizon)] for i in range(len(weights))]
    CM = [[model.NewIntVar(0, 1, f'CM_{i}_{t}') for t in range(horizon)] for i in range(len(weights))]
    SSD = [[model.NewIntVar(0, 1, f'SSD_{i}_{t}') for t in range(horizon)] for i in range(len(weights))]

    # Initialization
    for i in range(len(weights)):
        model.Add(SSD[i][0] == 1)
        for t in range(weights[i][4]):
            model.Add(CM[i][t] == 0)
        model.Add(GM[i][0] == 0)
        #for t in range(max([max(weight[1:]) for weight in weights])):
        #    model.Add(SSD[i][t] == 1)
        #    model.Add(GM[i][t] == 0)
        #    model.Add(CM[i][t] == 0)

    '''
    equal code for
        Load_C2G = [[
                sum(
                        [max(0, GM[i][j] - GM[i][j - 1]) for j in range(t + 1, t + weights[i][1] + 1)]
                    ) for t in range(horizon - weights[i][1])
                ] for i in range(len(weights))]
    '''
    Load_C2G = []
    for i in range(len(weights)):
        ele = []
        for t in range(horizon - weights[i][1]):
            tmp = []
            for j in range(t + 1, t + weights[i][1] + 1):
                act = model.NewIntVar(0, 1, f'Act_{i},{t},G')
                model.Add(act >= GM[i][j] - GM[i][j - 1])
                tmp.append(act)
            #ele.append(sum(tmp))
            e = model.NewIntVar(0, 1, f'Load_C2G_i{i}_t{t}')
            model.Add(sum(tmp) <= 1)
            model.Add(e == sum(tmp))
            ele.append(e)
        Load_C2G.append(ele)

    '''
    equal code for
    Load_C2SSD = [[
            sum(
                    [max(0, SSD[i][j] - SSD[i][j - 1]) for j in range(t + 1, t + weights[i][3] + 1)]
                ) for t in range(horizon - weights[i][3])
            ] for i in range(len(weights))]
    '''
    Load_C2SSD = []
    for i in range(len(weights)):
        ele = []
        for t in range(horizon - weights[i][3]):
            tmp = []
            for j in range(t + 1, t + weights[i][3] + 1):
                act = model.NewIntVar(0, 1, f'Act_{i},{t},SSD')
                model.Add(act >= SSD[i][j] - SSD[i][j - 1])
                tmp.append(act)
            e = model.NewIntVar(0, 1, f'Load_C2SSD_i{i}_t{t}')
            model.Add(sum(tmp) <= 1)
            model.Add(e == sum(tmp))
            ele.append(e)
        Load_C2SSD.append(ele)

    '''
    equal code for
    Act_C = [[
        max(0, CM[i][t] - CM[i][t - 1]) for t in range(1, horizon)
            ] for i in range(len(weights))]
    '''
    Act_C = []
    for i in range(len(weights)):
        ele = []
        for t in range(1, horizon):
            act = model.NewIntVar(0, 1, f'Act_{i},{t},C')
            model.Add(act >= CM[i][t] - CM[i][t - 1])
            ele.append(act)
        Act_C.append(ele)

    Act_G2C = [[model.NewIntVar(0, 1, f'ActG2C_{i}_{t}') for t in range(horizon - 1)] for i in range(len(weights))]
    Act_SSD2C = [[model.NewIntVar(0, 1, f'ActSSD2C_{i}_{t}') for t in range(horizon - 1)] for i in range(len(weights))]
    for i in range(len(weights)):
        for t in range(horizon - 1):
            model.Add(Act_G2C[i][t] + Act_SSD2C[i][t] == Act_C[i][t])
    #for i in range(len(weights)):
    #    for t in range(horizon - max(weights[i][2], weights[i][4]), horizon - min(weights[i][2], weights[i][4])):
    #        if weights[i][2] < weights[i][4]:
    #            model.Add(Act_G2C[i][t] == Act_C[i][t])
    #        if weights[i][4] < weights[i][2]:
    #            model.Add(Act_SSD2C[i][t] == Act_C[i][t])

    '''
    equal code for
    Load_G2C = [[
            sum(
                    [Act_G2C[i][j] for j in range(t, t + weights[i][2])]
                ) for t in range(horizon - weights[i][2])
            ] for i in range(len(weights))]
    '''
    Load_G2C = []
    for i in range(len(weights)):
        ele = []
        for t in range(horizon - weights[i][2]):
            e = model.NewIntVar(0, 1, f'Load_G2C_{i}_{t}')
            model.Add(sum([Act_G2C[i][j] for j in range(t, t + weights[i][2])]) <= 1)
            model.Add(e == sum([Act_G2C[i][j] for j in range(t, t + weights[i][2])]))
            ele.append(e)
        Load_G2C.append(ele)

    '''
    equal code for
    Load_SSD2C = [[
            sum(
                    [Act_SSD2C[i][j] for j in range(t, t + weights[i][4])]
                ) for t in range(horizon - weights[i][4])
            ] for i in range(len(weights))]
    '''
    Load_SSD2C = []
    for i in range(len(weights)):
        ele = []
        for t in range(horizon - weights[i][4]):
            e = model.NewIntVar(0, 1, f'Load_SSD2C_{i}_{t}')
            model.Add(sum([Act_SSD2C[i][j] for j in range(t, t + weights[i][4])]) <= 1)
            model.Add(e == sum([Act_SSD2C[i][j] for j in range(t, t + weights[i][4])]))
            ele.append(e)
        Load_SSD2C.append(ele)

    # G2C and C2G considered later
    for t in range(horizon):
        Pcie_C2SSD = []
        Pcie_SSD2C = []
        for i in range(len(weights)):
            if t < horizon - weights[i][3]:
                Pcie_C2SSD.append(Load_C2SSD[i][t])
            if t < horizon - weights[i][4]:
                Pcie_SSD2C.append(Load_SSD2C[i][t])
        model.Add(sum(Pcie_C2SSD) <= 1)
        model.Add(sum(Pcie_SSD2C) <= 1)

    # Must exist when loading/offloading
    for i in range(len(weights)):
        for t in range(horizon - weights[i][1]):
            model.Add(Load_C2G[i][t] <= CM[i][t])
    for i in range(len(weights)):
        for t in range(horizon - weights[i][2]):
            model.Add(Load_G2C[i][t] <= GM[i][t])
    for i in range(len(weights)):
        for t in range(horizon - weights[i][3]):
            model.Add(Load_C2SSD[i][t] <= CM[i][t])
    for i in range(len(weights)):
        for t in range(horizon - weights[i][4]):
            model.Add(Load_SSD2C[i][t] <= SSD[i][t])

    dev = [model.NewIntVar(0, 1, f'dev{i}') for i in range(num_tasks)]
    task_starts = [model.NewIntVar(0, horizon, f'task{i}start') for i in range(num_tasks)]
    task_ends = [model.NewIntVar(0, horizon, f'task{i}end') for i in range(num_tasks)]
    task_intervals = [model.NewIntervalVar(task_starts[i], dev[i] * tasks[i][1][1] + (1 - dev[i]) * tasks[i][1][0], task_ends[i], f'task_interval{i}')
        for i in range(num_tasks)]

    weight_loaded_G = [[
        model.NewFixedSizeIntervalVar(t, 1, f'weight_loaded_G_{i}{t}') for t in range(horizon)
    ] for i in range(num_tasks)]
    weight_loaded_C = [[
        model.NewFixedSizeIntervalVar(t, 1, f'weight_loaded_C_{i}{t}') for t in range(horizon)
    ] for i in range(num_tasks)]

    # Must have all weights and tensors on device
    GM_minus = [[model.NewIntVar(0, 1, f'GM_minus_{i}_{t}') for t in range(horizon)] for i in range(len(weights))]
    CM_minus = [[model.NewIntVar(0, 1, f'CM_minus_{i}_{t}') for t in range(horizon)] for i in range(len(weights))]
    for i in range(len(weights)):
        for t in range(horizon):
            model.Add(GM_minus[i][t] == 1-GM[i][t])
            model.Add(CM_minus[i][t] == 1-CM[i][t])
    for i in range(num_tasks):
        model.AddCumulative(weight_loaded_G[i] + [task_intervals[i]], [GM_minus[tasks[i][3]][t] for t in range(horizon)] + [dev[i]], 1)
    for i in range(num_tasks):
        model.AddCumulative(weight_loaded_C[i] + [task_intervals[i]], [CM_minus[tasks[i][3]][t] for t in range(horizon)] + [1-dev[i]], 1)
    
    print(model.Validate())

    # output mem map
    # node -> (Duration, GM, CM)
    aux_interval_mem = {}
    # (task_ij_interval, pcie_usage)
    ij_C2G_Pcie = []
    ij_G2C_Pcie = []
    for j, _, predecessors, _, _, duration in tasks:
        for i in predecessors:
            # start/end of node ij that transfers outputs of node i
            task_ij_start = model.NewIntVar(0, horizon, f'taskij{i}{j}start')
            task_ij_end = model.NewIntVar(0, horizon, f'taskij{i}{j}end')

            # devij: if dev[i] == 1 and dev[j] == 0. 
            # similar for devji 
            devij = model.NewBoolVar('devi{i}j{j}')
            devji = model.NewBoolVar('devi{j}j{i}')
            model.Add(dev[i] > dev[j]).OnlyEnforceIf(devij)
            model.Add(dev[i] <= dev[j]).OnlyEnforceIf(devij.Not())
            model.Add(dev[j] > dev[i]).OnlyEnforceIf(devji)
            model.Add(dev[j] <= dev[i]).OnlyEnforceIf(devji.Not())
            task_ij_interval = model.NewIntervalVar(
                    task_ij_start, 
                    duration,
                    task_ij_end, 
                    f'ij_interval{i}'
            )

            # Pcie
            tmp = model.NewIntVar(0, 1, f'max(dev[j{j}] - dev[i{i}], 0)')
            model.Add(tmp == 1).OnlyEnforceIf(devji)
            model.Add(tmp == 0).OnlyEnforceIf(devji.Not())
            ij_C2G_Pcie.append((task_ij_interval, tmp))

            tmp = model.NewIntVar(0, 1, f'max(dev[i{i}] - dev[j{j}], 0)')
            model.Add(tmp == 1).OnlyEnforceIf(devij)
            model.Add(tmp == 0).OnlyEnforceIf(devij.Not())
            ij_G2C_Pcie.append((task_ij_interval, tmp))

        # Dependency
            model.Add(task_ends[i] <= task_ij_start)
            model.Add(task_ij_end <= task_starts[j])

            # Tensor lifecycle for outputs of node i
            tmp = model.NewIntVar(0, horizon, f'mem_i_ij_interval_duration_i{i}_ij{i}_{j}')
            model.Add(tmp == task_ij_end - task_starts[i])
            mem_i_ij_interval = model.NewIntervalVar(task_starts[i], tmp, task_ij_end, f'mem_{i}_{i}{j}')

            tmp = model.NewIntVar(0, horizon, f'mem_ij_j_interval_duration_ij{i}_{j}_j{j}')
            model.Add(tmp == task_ends[j] - task_ij_end)
            mem_ij_j_interval = model.NewIntervalVar(task_ij_end, tmp, task_ends[j], f'mem_{i}{j}_{j}')

            tmp1 = model.NewIntVar(0, tasks[i][4], f'aux_interval_mem_j{j}_1')
            tmp2 = model.NewIntVar(0, tasks[i][4], f'aux_interval_mem_j{j}_2')
            model.Add(tmp1 == dev[i] * tasks[i][4])
            model.Add(tmp2 == (1 - dev[i]) * tasks[i][4])
            aux_interval_mem[f'i{i}ij{i} {j}'] = (mem_i_ij_interval, tmp1, tmp2)

            tmp1 = model.NewIntVar(0, tasks[i][4], f'aux_interval_mem_i{i}j{j}_1')
            tmp2 = model.NewIntVar(0, tasks[i][4], f'aux_interval_mem_i{i}j{j}_2')
            model.Add(tmp1 == dev[j] * tasks[i][4])
            model.Add(tmp2 == (1 - dev[j]) * tasks[i][4])
            aux_interval_mem[f'ij{i} {j}j{j}'] = (mem_ij_j_interval, tmp1, tmp2)

    aux_interval_C2G = [model.NewFixedSizeIntervalVar(t, 1, f'aux_interval_C2G_{i}{t}') for i in range(len(weights)) for t in range(horizon - weights[i][1])]
    aux_interval_G2C = [model.NewFixedSizeIntervalVar(t, 1, f'aux_interval_G2C_{i}{t}') for i in range(len(weights)) for t in range(horizon - weights[i][2])]
    aux_pcie_C2G = [Load_C2G[i][t] for i in range(len(weights)) for t in range(horizon - weights[i][1])]
    aux_pcie_G2C = [Load_G2C[i][t] for i in range(len(weights)) for t in range(horizon - weights[i][2])]

    #PCIE of C2G and G2C
    model.AddCumulative(aux_interval_C2G + [task_ij_interval for (task_ij_interval, _) in ij_C2G_Pcie], aux_pcie_C2G + [pcie_usage for (_, pcie_usage) in ij_C2G_Pcie], 1)
    model.AddCumulative(aux_interval_G2C + [task_ij_interval for (task_ij_interval, _) in ij_G2C_Pcie], aux_pcie_G2C + [pcie_usage for (_, pcie_usage) in ij_G2C_Pcie], 1)
    
    interval_mem = [
        model.NewFixedSizeIntervalVar(t, 1, f'interval_mem_{t}') for t in range(horizon)
    ]

    '''
    equal code for
    interval_gmem_usage = [sum([GM[i][t] for i in range(len(weights))]) for t in range(horizon)]
    interval_cmem_usage = [sum([CM[i][t] for i in range(len(weights))]) for t in range(horizon)]
    '''
    interval_gmem_usage = []
    interval_cmem_usage = []
    for t in range(horizon):
        ge = model.NewIntVar(0, sum([weight[0] for weight in weights]), f'interval_gmem_usage_t')
        ce = model.NewIntVar(0, sum([weight[0] for weight in weights]), f'interval_cmem_usage_t')
        model.Add(ge == sum([GM[i][t] * weights[i][0] for i in range(len(weights))]))
        model.Add(ce == sum([CM[i][t] * weights[i][0] for i in range(len(weights))]))
        interval_gmem_usage.append(ge)
        interval_cmem_usage.append(ce)

    interval_mem_C2G = [
        model.NewFixedSizeIntervalVar(t, 1, f'interval_mem_C2G_i{i}_t{t}') for i in range(len(weights)) for t in range(horizon - weights[i][1])
    ]
    interval_mem_usage_C2G = [Load_C2G[i][t] * weights[i][0] for i in range(len(weights)) for t in range(horizon - weights[i][1])]

    interval_mem_G2C = [
        model.NewFixedSizeIntervalVar(t, 1, f'interval_mem_G2C_i{i}_t{t}') for i in range(len(weights)) for t in range(horizon - weights[i][2])
    ]
    interval_mem_usage_G2C = [Load_G2C[i][t] * weights[i][0] for i in range(len(weights)) for t in range(horizon - weights[i][2])]

    interval_mem_C2SSD = [
        model.NewFixedSizeIntervalVar(t, 1, f'interval_mem_C2SSD_i{i}_t{t}') for i in range(len(weights)) for t in range(horizon - weights[i][3])
    ]
    interval_mem_usage_C2SSD = [Load_C2SSD[i][t] * weights[i][0] for i in range(len(weights)) for t in range(horizon - weights[i][3])]

    interval_mem_SSD2C = [
        model.NewFixedSizeIntervalVar(t, 1, f'interval_mem_SSD2C_i{i}_t{t}') for i in range(len(weights)) for t in range(horizon - weights[i][4])
    ]
    interval_mem_usage_SSD2C = [Load_SSD2C[i][t] * weights[i][0] for i in range(len(weights)) for t in range(horizon - weights[i][4])]

    # Memory Constraint
    model.AddCumulative(
        interval_mem + interval_mem_C2G + interval_mem_G2C + [aux_interval for _, (aux_interval, _, _) in aux_interval_mem.items()],
        interval_gmem_usage + interval_mem_usage_C2G + interval_mem_usage_G2C + [aux_interval_gpum for _, (_, aux_interval_gpum, _) in aux_interval_mem.items()],
        GPUM,
    )

    model.AddCumulative(
        interval_mem + interval_mem_C2G + interval_mem_G2C + interval_mem_C2SSD + interval_mem_SSD2C + [aux_interval for _, (aux_interval, _, _) in aux_interval_mem.items()],
        interval_cmem_usage + interval_mem_usage_C2G + interval_mem_usage_G2C + interval_mem_usage_C2SSD + interval_mem_usage_SSD2C + [aux_interval_cpum for _, (_, _, aux_interval_cpum) in aux_interval_mem.items()],
        CPUM,
    )

    # Resource constraints
    #num_resources = len(resource_capacities)
    #for i, capacity in enumerate(resource_capacities):
    #    model.AddCumulative(task_intervals, [task[3][i] for task in data], capacity)
    
    model.Minimize(task_ends[-1])
    
    solver = cp_model.CpSolver()
    printer = cp_model.VarArraySolutionPrinter(dev + task_ends)
    status = solver.Solve(model, printer)
    print(status)
    
    #res = []
    #for i in range(len(data)):
    #    res.append(solver.Value(task_starts[i]))
    #print(np.argsort(res))
    print(cp_model.MODEL_INVALID)
    print(cp_model.FEASIBLE)
    print(cp_model.INFEASIBLE)
    print(cp_model.OPTIMAL)


if __name__ == '__main__':
    # GPUM, CPUM, SSDM
    weights = [
        # (Size, C->G, G->C, C->SSD, SSD->C)
        (40, 4, 4, 12, 12),
        (30, 3, 3, 9, 9),
        (20, 2, 2, 6, 6),
    ]
    CPU_GPU_RATIO = 3
    tasks = [
        # ID, Duration(CPU, GPU), Dependencies, WeightIndex(s), Output Mem, Output Tranfer Time(C->G == G->C, currently)
        (0, (int(5 * CPU_GPU_RATIO), 5), [], 0, 5, 1),
        (1, (int(7 * CPU_GPU_RATIO), 7), [0], 1, 10, 2),
        (2, (int(8 * CPU_GPU_RATIO), 8), [0], 2, 5, 1),
        (3, (int(2 * CPU_GPU_RATIO), 2), [1, 2], 0, 5, 1),
        (4, (int(4 * CPU_GPU_RATIO), 4), [1, 3], 2, 10, 2),
    ]
    #resource_capacities = [3, 3, 3]
    mp_general(tasks, weights)

