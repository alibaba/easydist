import argparse

import numpy as np
import torch
from easydist.torch import tensorfield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnk', default=4096, type=int, help='Matrix Size (N x N)')
    parser.add_argument('--precision',
                        default='fp16',
                        type=str,
                        help='Precision (fp16, fp32)',
                        choices=['fp16', 'fp32'])
    parser.add_argument('--trials', default=10, type=int, help='Number of Trials to Execute')
    parser.add_argument('--warmup-trials', default=5, type=int, help='Warmup Trials to discard')
    parser.add_argument('--tensorfield', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()

    if args.tensorfield:
        tensorfield.init_tensorfield_allocator()
        print("Tensorfield allocator initialized.")

    if args.profile:
        log_dir = f'./log/matmul_tfield_{args.tensorfield}_profile'
        prof = torch.profiler.profile(schedule=torch.profiler.schedule(wait=1,
                                                                       warmup=args.warmup_trials,
                                                                       active=args.trials,
                                                                       repeat=1),
                                      on_trace_ready=torch.profiler.tensorboard_trace_handler(
                                          log_dir, use_gzip=True),
                                      profile_memory=False,
                                      record_shapes=False,
                                      with_stack=False)
        prof.start()

    start_evt, end_evt = [], []
    for _ in range(0, args.trials):
        start_evt.append(torch.cuda.Event(enable_timing=True))
        end_evt.append(torch.cuda.Event(enable_timing=True))

    for trial in range(0, args.trials + args.warmup_trials):
        evt_idx = trial - args.warmup_trials

        if evt_idx >= 0:
            start_evt[evt_idx].record()

        precision = torch.float32 if args.precision == 'fp32' else torch.float16

        tensor1 = torch.rand(args.mnk, args.mnk, device='cuda', dtype=precision)
        tensor2 = torch.rand(args.mnk, args.mnk, device='cuda', dtype=precision)
        _ = torch.mm(tensor1, tensor2)

        if evt_idx >= 0:
            end_evt[evt_idx].record()

        if args.profile:
            prof.step()

    torch.cuda.synchronize()

    if args.profile:
        prof.stop()

    elapsed_time_ms = np.zeros(args.trials)
    for trial in range(0, args.trials):
        elapsed_time_ms[trial] = start_evt[trial].elapsed_time(end_evt[trial])

    print(
        f"Average time elapsed: {np.mean(elapsed_time_ms)} ms (variance: {np.var(elapsed_time_ms)}"
    )

    if args.tensorfield:
        tensorfield.finalize_tensorfield_allocator()


if __name__ == '__main__':
    main()
