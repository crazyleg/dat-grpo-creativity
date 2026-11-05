"""Run one isolated job with a host-memory watchdog independent of CUDA/PyTorch."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':',1)
        values[key] = int(value.split()[0])*1024
    return values['MemAvailable'], values['SwapTotal']-values['SwapFree']


def memory_stop_reason(available,swap_growth,min_available_gib,soft_swap_gib):
    if available < min_available_gib*2**30:
        return 'host_memory_floor'
    if swap_growth > soft_swap_gib*2**30 and available < 26*2**30:
        return 'swap_growth'
    if swap_growth > 4*2**30:
        return 'excessive_swap_growth'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--min-available-gib', type=float, default=18)
    parser.add_argument('--max-new-swap-gib', type=float, default=.25)
    parser.add_argument('--timeout-seconds', type=float, default=1800)
    parser.add_argument('--progress-timeout-seconds', type=float, default=1200)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A child command is required')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    available, initial_swap = memory()
    if available < args.min_available_gib*2**30:
        raise SystemExit('Insufficient host headroom to launch')
    environment = dict(os.environ, DAT_GRPO_WATCHDOG_PID=str(os.getpid()))
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True, env=environment)
    (args.run_dir/'worker-pid').write_text(str(child.pid)+'\n')
    started = time.monotonic()
    report = dict(worker_pid=child.pid, minimum_available_gib=available/2**30,
                  peak_swap_gib=initial_swap/2**30, memory_floor_gib=args.min_available_gib,
                  soft_swap_growth_gib=args.max_new_swap_gib,swap_headroom_limit_gib=26,
                  hard_swap_growth_gib=4,progress_timeout_seconds=args.progress_timeout_seconds,
                  wall_time_limit_seconds=args.timeout_seconds)
    reason = None
    last_report = 0
    last_progress = started
    previous_progress = None
    try:
        while child.poll() is None:
            available, swap = memory()
            report['minimum_available_gib'] = min(report['minimum_available_gib'],available/2**30)
            report['peak_swap_gib'] = max(report['peak_swap_gib'],swap/2**30)
            elapsed = time.monotonic()-started
            status_path = args.run_dir/'status.json'
            try:
                status = json.loads(status_path.read_text())
                progress = (status.get('status'),status.get('optimizer_step'),status.get('last_checkpoint_step'))
                if progress != previous_progress:
                    previous_progress = progress
                    last_progress = time.monotonic()
            except (FileNotFoundError,json.JSONDecodeError):
                pass
            reason = memory_stop_reason(available,swap-initial_swap,args.min_available_gib,args.max_new_swap_gib)
            if reason is None and elapsed > args.timeout_seconds:
                reason = 'wall_time_limit'
            elif reason is None and time.monotonic()-last_progress > args.progress_timeout_seconds:
                reason = 'no_training_progress'
            if elapsed-last_report >= 2 or reason:
                report.update(elapsed_seconds=elapsed,reason=reason,alive=child.poll() is None)
                (args.run_dir/'watchdog.json').write_text(json.dumps(report,indent=2)+'\n')
                last_report = elapsed
            if reason:
                break
            time.sleep(.2)
    except KeyboardInterrupt:
        reason = 'user_interrupt'
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGINT if reason=='user_interrupt' else signal.SIGTERM)
            try:
                child.wait(timeout=15 if reason=='user_interrupt' else 2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=10)
        report.update(reason=reason,alive=False,exit_code=child.returncode,elapsed_seconds=time.monotonic()-started)
        (args.run_dir/'watchdog.json').write_text(json.dumps(report,indent=2)+'\n')
        history=args.run_dir/'watchdog-history'
        history.mkdir(exist_ok=True)
        (history/f'{int(time.time())}-{child.pid}.json').write_text(json.dumps(report,indent=2)+'\n')
        if reason:
            path = args.run_dir/'status.json'
            status = json.loads(path.read_text()) if path.exists() else {}
            status.update(status='interrupted' if reason=='user_interrupt' else 'failed',failure_reason='watchdog_'+reason,
                restart_blocked=reason!='user_interrupt')
            path.write_text(json.dumps(status,indent=2)+'\n')
            if reason!='user_interrupt':
                (args.run_dir/'restart-blocked.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(1 if reason else child.returncode)


if __name__ == '__main__':
    main()
