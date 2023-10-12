"""Minimal jobshop example."""
import collections
from ortools.sat.python import cp_model

MODE_COMM = 0
MODE_COMP = 1

'''
jobs_data = [task = (machine_id, processing_time),]
dependencies = [dependency = ((job_id, task_id), (job_id, task_id)),]
'''
def RCPSP(jobs_data, dependencies):
    
    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    
    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Create the model.
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple("task_type", "start end interval")
    
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple(
        "assigned_task_type", "start job index duration"
    )

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine, duration = task
            suffix = f"_{job_id}_{task_id}"
            start_var = model.NewIntVar(0, horizon, "start" + suffix)
            end_var = model.NewIntVar(0, horizon, "end" + suffix)
            interval_var = model.NewIntervalVar(
                start_var, duration, end_var, "interval" + suffix
            )
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var
            )
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for (pre_job_id, pre_task_id), (suc_job_id, suc_task_id) in dependencies:
        model.Add(all_tasks[pre_job_id, pre_task_id].end <= 
                  all_tasks[suc_job_id, suc_task_id].start)

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
                    )
                )

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