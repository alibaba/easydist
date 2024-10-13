# Copyright (c) 2023, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import collections
import logging
import numpy as np
from ortools.sat.python import cp_model

from itertools import product
from mip import Model, xsum, BINARY

import easydist.config as mdconfig

logger = logging.getLogger(__name__)


def rcpsp_genetic(num_tasks, processing_time, resource_consumption, available_resources,
                  precedence_relations):
    """
    This function solves the Resource-Constrained Project Scheduling Problem (RCPSP) using the python-mip library.

    Args:
    num_tasks (int): The number of tasks plus the two 'dummy' ones.
    processing_time (list): The processing time of each task.
    resource_consumption (list): The resource consumption of each task.
    available_resources (list): The resources available.
    precedence_relations (list): The precedence relations.

    Returns:
    None
    """

    # Create ranges for resources, tasks, and time periods
    resources_range = range(len(available_resources))
    tasks_range = range(len(processing_time))
    time_period_range = range(sum(processing_time))

    # Create the model
    model = Model()

    # Create the decision variables
    decision_variables = [[
        model.add_var(name=f"x({task},{time_period})", var_type=BINARY)
        for time_period in time_period_range
    ] for task in tasks_range]

    # Set the objective function
    model.objective = xsum(time_period * decision_variables[num_tasks + 1][time_period]
                           for time_period in time_period_range)

    # Set the constraints
    for task in tasks_range:
        model += xsum(decision_variables[task][time_period]
                      for time_period in time_period_range) == 1

    for (resource, time_period) in product(resources_range, time_period_range):
        model += (xsum(
            resource_consumption[task][resource] * decision_variables[task][time_period_2]
            for task in tasks_range
            for time_period_2 in range(max(0, time_period - processing_time[task] +
                                           1), time_period + 1)) <= available_resources[resource])

    for (task, successor) in precedence_relations:
        model += xsum(time_period * decision_variables[successor][time_period] -
                      time_period * decision_variables[task][time_period]
                      for time_period in time_period_range) >= processing_time[task]

    # Optimize the model
    model.optimize()

    # Print the optimal schedule
    for (task, time_period) in product(tasks_range, time_period_range):
        if decision_variables[task][time_period].x >= 0.99:
            print(
                f"task {task}: begins at t={time_period} and finishes at t={time_period+processing_time[task]}"
            )


def rcpsp_general(task_data, resource_capacities, dep_rec_mask=None):
    '''
    This function solves the Resource-Constrained Project Scheduling Problem (RCPSP) using or-tools from Google.

    Args:
    task_data: [(task_id(unique), duration, predecessor, 
                 independent_resource_usage, dependent_resource_usage, priority)]
    available_resources (list): The resources available.

    Returns:
    An ordering of index representing the scheduled order
    '''

    model = cp_model.CpModel()
    num_tasks = len(task_data)
    horizon = sum(task[1] for task in task_data)
    makespan = model.NewIntVar(0, horizon, 'makespan')

    task_starts = [model.NewIntVar(0, horizon, f'task{i}start') for i in range(num_tasks)]
    task_ends = [model.NewIntVar(0, horizon, f'task{i}end') for i in range(num_tasks)]
    task_intervals = [
        model.NewIntervalVar(task_starts[i], task_data[i][1], task_ends[i], f'interval{i}')
        for i in range(num_tasks)
    ]

    for task, _, predecessors, _, _ in task_data:
        for predecessor in predecessors:
            model.Add(task_ends[predecessor] <= task_starts[task])

    dependent_rec = False
    for i in dep_rec_mask:
        if i == 1:
            dependent_rec = True
            break
    if dependent_rec:
        task_dep_ends = [model.NewIntVar(0, horizon, f'task_dep{i}end') for i in range(num_tasks)]
        task_dep_durations = [
            model.NewIntVar(0, horizon, f'task_dep{i}end') for i in range(num_tasks)
        ]

        successors = [[i] for i in range(num_tasks)]
        for i, (_, _, predecessors, _, _) in enumerate(task_data):
            for predecessor in predecessors:
                successors[predecessor].append(i)

        for i, task_dep_end in enumerate(task_dep_ends):
            model.AddMaxEquality(task_dep_end, [task_ends[j] for j in successors[i]])
            model.Add(task_dep_durations[i] == task_dep_end - task_starts[i])

        task_dep_intervals = [
            model.NewIntervalVar(task_starts[i], task_dep_durations[i], task_dep_ends[i],
                                 f'dep_interval{i}') for i in range(num_tasks)
        ]

    # Resource constraints
    for i, capacity in enumerate(resource_capacities):
        if dep_rec_mask is not None and dep_rec_mask[i] == 1:
            model.AddCumulative(task_dep_intervals, [task[3][i] for task in task_data], capacity)
        else:
            model.AddCumulative(task_intervals, [task[3][i] for task in task_data], capacity)

    model.Minimize(task_ends[-1])

    solver = cp_model.CpSolver()
    printer = cp_model.VarArraySolutionPrinter(task_starts)
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        priorities = [pri for (_, _, _ ,_ , pri) in task_data]
        res_starts = []
        res_ends = []
        for i in range(len(task_data)):
            res_starts.append(solver.Value(task_starts[i]))
            res_ends.append(solver.Value(task_ends[i]))
        ori_order = np.argsort(res_starts)
        
        selected = [False] * num_tasks
        res = []
        idx = 0

        while len(res) < num_tasks:
            if selected[ori_order[idx]]:
                idx += 1
                continue
            cur_end = res_ends[ori_order[idx]]
            target_idx = idx
            for next_idx in range(idx + 1, num_tasks):
                if res_starts[ori_order[next_idx]] >= cur_end:
                    break
                if not selected[ori_order[next_idx]] and priorities[ori_order[next_idx]] \
                                            > priorities[ori_order[target_idx]]:
                    target_idx = next_idx
            selected[ori_order[target_idx]] = True
            res.append(ori_order[target_idx])

        return res
    else:
        raise RuntimeError("RCPSP: No solution found!")


def rcpsp_data_select(task_data, select_mask):

    raw_data_selected = [task_data[i] for i in range(len(task_data)) if select_mask[i] == 1]
    keys_selected = [raw_data[0] for raw_data in raw_data_selected]

    # dependency redundancies removal
    data_selected = [(key, duration, [dp for dp in dependencies if dp in keys_selected], resource_uses, priority)
                        for (key, duration, dependencies, resource_uses, priority) in raw_data_selected]

    return data_selected


def rcpsp_reorder(task_data, select_mask, schedule):

    task_data_buffer = [task_data[i] for i in range(len(task_data)) if select_mask[i] == 1]
    assert(len(task_data_buffer) == len(schedule))

    idx = 0
    for i, mask in enumerate(select_mask):
        if mask == 1:
            task_data[i] = task_data_buffer[schedule[idx]]
            idx += 1
    return task_data


def rcpsp_data_transform(task_data, resource_to_id, resource_num):
    transformed_task_data = []
    key_to_id = {}
    for i, (task_key, duration, dependencies, resource_uses, priority) in enumerate(task_data):
        key_to_id[task_key] = i
        resource_in_use = [0] * resource_num
        for r, amount in resource_uses:
            resource_in_use[resource_to_id[r]] = amount
        transformed_task_data.append((i, duration, 
                                      [key_to_id[dp] for dp in dependencies], 
                                      resource_in_use, priority))
    return transformed_task_data


def rcpsp(task_data, available_resources, rec_dep_mask, method):
    '''
    Main entry of the solver.

    Args:
    task_data: [(task_key, duration, predecessor, resource_usage)]
    available_resources (dict): a mapping from resources to its available amount
    rec_dep_mask: denoting whether recouce live along dependencies
    method: how to perform rcpsp scheduling

    Returns:
    An ordering of index representing the scheduled order
    '''

    #TODO check if input is legal

    resource_capacities = []
    resource_to_id = {}
    resource_num = len(available_resources)
    for i, resource in enumerate(available_resources):
        resource_to_id[resource] = i
        resource_capacities.append(available_resources[resource])


    if method == 'general':

        transformed_task_data = rcpsp_data_transform(task_data, resource_to_id, resource_num)

        schedule = rcpsp_general(transformed_task_data, resource_capacities, 
                                 rec_dep_mask)

    elif method == 'genetic':

        transformed_task_data = rcpsp_data_transform(task_data, resource_to_id, resource_num)

        num_tasks = len(task_data)
        processing_time = [0] * len(num_tasks)
        resource_consumption = [[]] * len(num_tasks)
        precedence_relations = [[]] * len(num_tasks)
        for (task_id, duration, dependencies, resource_in_use) in transformed_task_data:
            processing_time[task_id] = duration
            resource_consumption[task_id] = resource_in_use
            precedence_relations[task_id] = dependencies

        schedule = rcpsp_genetic(num_tasks, processing_time, resource_consumption,
                                 resource_capacities, precedence_relations)

    elif method == 'odd_even':
        _task_data = [task for task in task_data]
        num_tasks = len(_task_data)
        schedule = [i for i in range(num_tasks)]
        
        block_size = 125
        odd_oven_points = [i * block_size for i in range(int(num_tasks / block_size))]
        odd_oven_points.append(num_tasks - 1)
        # block: (start, end)
        blocks = [(odd_oven_points[i], odd_oven_points[i + 1]) for i in range(len(odd_oven_points) - 1)]
        block_num = len(blocks)

        fused_adam_task_idx = -1
        for idx, task_d in enumerate(_task_data):
            if "_fused_adam" in task_d[0].target.__str__():
                fused_adam_task_idx = idx
                continue

        for _ in range(mdconfig.rcpsp_iter_round):
            # odd round
            # [block0, block1], [block2, block3] ...
            for b in range(0, block_num, 2):
                if b + 1 >= block_num:
                    break
                
                # can be extended to support random node selection
                select_mask = np.array([0] * num_tasks)
                for i in range(blocks[b][0], blocks[b + 1][1] + 1):
                    select_mask[i] = 1
                select_mask[fused_adam_task_idx:] = 0

                if select_mask.sum() == 0:
                    continue

                data_selected = rcpsp_data_select(_task_data, select_mask)
                transformed_task_data = rcpsp_data_transform(data_selected, resource_to_id, resource_num)

                _schedule = rcpsp_general(transformed_task_data, resource_capacities, rec_dep_mask)
                
                _task_data = rcpsp_reorder(_task_data, select_mask, _schedule)
                schedule = rcpsp_reorder(schedule, select_mask, _schedule)

            # even round
            # [block1, block2], [block3, block4] ...
            for b in range(1, block_num, 2):
                if b + 1 >= block_num:
                    break

                select_mask = np.array([0] * num_tasks)
                for i in range(blocks[b][0], blocks[b + 1][1] + 1):
                    select_mask[i] = 1
                select_mask[fused_adam_task_idx:] = 0

                if select_mask.sum() == 0:
                    continue

                data_selected = rcpsp_data_select(_task_data, select_mask)
                transformed_task_data = rcpsp_data_transform(data_selected, resource_to_id, resource_num)

                _schedule = rcpsp_general(transformed_task_data, resource_capacities, rec_dep_mask)
                
                _task_data = rcpsp_reorder(_task_data, select_mask, _schedule)
                schedule = rcpsp_reorder(schedule, select_mask, _schedule)

    logger.info("RCPSP: Optimal found")
    return schedule


MODE_COMM = 0
MODE_COMP = 1
'''
jobs_data = [task = (machine_id, processing_time),]
dependencies = [dependency = ((job_id, task_id), (job_id, task_id)),]
'''


def rcpsp_jobshop(jobs_data, dependencies):

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Create the model.
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple("task_type", "start end interval")

    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple("assigned_task_type", "start job index duration")

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine, duration = task
            suffix = f"_{job_id}_{task_id}"
            start_var = model.NewIntVar(0, horizon, "start" + suffix)
            end_var = model.NewIntVar(0, horizon, "end" + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var, "interval" + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var,
                                                   end=end_var,
                                                   interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for (pre_job_id, pre_task_id), (suc_job_id, suc_task_id) in dependencies:
        model.Add(all_tasks[pre_job_id, pre_task_id].end <= all_tasks[suc_job_id,
                                                                      suc_task_id].start)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        obj_var,
        [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)],
    )
    model.Minimize(obj_var)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=solver.Value(all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1],
                    ))

    assigned_jobs[all_machines[MODE_COMM]].sort()
    assigned_jobs[all_machines[MODE_COMP]].sort()
    idx_0 = 0
    idx_1 = 0
    schedule = []

    while idx_0 < len(assigned_jobs[MODE_COMM]) and \
        idx_1 < len(assigned_jobs[MODE_COMP]):
        if assigned_jobs[MODE_COMM][idx_0].start <= assigned_jobs[MODE_COMP][idx_1].start:
            schedule.append((MODE_COMM, assigned_jobs[MODE_COMM][idx_0].job))
            idx_0 += 1
        else:
            schedule.append((MODE_COMP, assigned_jobs[MODE_COMP][idx_1].job))
            idx_1 += 1

    if idx_0 < len(assigned_jobs[MODE_COMM]):
        for j in assigned_jobs[MODE_COMM][idx_0:]:
            schedule.append((MODE_COMM, j.job))
    else:
        for j in assigned_jobs[MODE_COMP][idx_1:]:
            schedule.append((MODE_COMP, j.job))

    return schedule
