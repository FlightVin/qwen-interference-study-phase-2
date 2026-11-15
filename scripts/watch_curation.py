"""Laptop-side watcher for the Babel curation jobs (read-only; no agent, no tokens while quiet).

Polls Babel over ssh every --interval seconds. It reads the state of the arnav-drl-* jobs and the
workers' PROGRESS lines, then EXITS with one line per event when something needs a look:
  JOB <name> <old> -> <new>        a job changed state (started, finished, failed, requeued)
  PROGRESS <p>% ...               overall progress crossed a 5% mark
  LOGERR <file>: <line>           a new error-like line in a job log
  SSH <message>                   Babel unreachable 3 polls in a row (log in again)
It reports THAT something happened, never what to do. A watermark file stops repeats.

Overall progress (plan doc, stage weights): 5% + 55% x (progress of hard screening toward
70 qualified problems, or toward screening all candidates, whichever is further).

Usage: python3 scripts/watch_curation.py --wm <watermark.json> [--interval 120]
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

TARGET, CANDIDATES = 70, 1961
REMOTE = r"""
squeue -h -u $USER -o '%i|%j|%T|%R' | grep '|arnav-drl-' || true
echo '---SACCT'
sacct -n -X -P -S now-2days --name=arnav-drl-env,arnav-drl-cur-g0,arnav-drl-cur-g1,arnav-drl-cur-p0,arnav-drl-cur-p1,arnav-drl-cur-p2,arnav-drl-cur-p3,arnav-drl-collect -o JobID,JobName,State 2>/dev/null || true
echo '---PROGRESS'
grep -h 'PROGRESS qualified=' ~/arnav/drl-logs/arnav-drl-cur-*.out 2>/dev/null | tail -n 200 || true
echo '---ERRORS'
grep -HnE 'Traceback|^[A-Za-z]*Error:|ERROR|Killed|oom|CUDA out of memory|exit code [1-9]|NOT complete' ~/arnav/drl-logs/arnav-drl-*.out 2>/dev/null | tail -n 50 || true
"""


def poll() -> str:
    r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "babel", REMOTE],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0 and "---SACCT" not in r.stdout:
        raise RuntimeError(r.stderr.strip()[:200])
    return r.stdout


def parse(out: str):
    sq, rest = out.split("---SACCT", 1)
    sa, rest = rest.split("---PROGRESS", 1)
    pr, er = rest.split("---ERRORS", 1)
    states = {}
    for line in sa.strip().splitlines():  # finished or running, from accounting
        jid, name, st = (line.split("|") + ["", "", ""])[:3]
        if name.startswith("arnav-drl-"):
            states[name] = st.split()[0]
    for line in sq.strip().splitlines():  # live queue wins (PENDING/RUNNING with reason)
        jid, name, st, reason = (line.split("|") + ["", "", "", ""])[:4]
        states[name] = st if st != "PENDING" else f"PENDING({reason.strip('()')})"
    q = f = 0
    for line in pr.splitlines():
        m = re.search(r"qualified=(\d+)/\d+ finished=(\d+)", line)
        if m:
            q, f = max(q, int(m.group(1))), max(f, int(m.group(2)))
    errors = [l for l in er.strip().splitlines() if l]
    return states, q, f, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--max-hours", type=float, default=12)
    a = ap.parse_args()
    wm_path = Path(a.wm)
    wm = json.loads(wm_path.read_text()) if wm_path.exists() else {"states": {}, "mark": 0, "errors": []}
    fails, t_end = 0, time.time() + a.max_hours * 3600
    while time.time() < t_end:
        try:
            states, q, f, errors = parse(poll())
            fails = 0
        except Exception as e:  # network or ssh master gone
            fails += 1
            if fails >= 3:
                print(f"SSH Babel unreachable 3 times in a row: {e}", flush=True)
                return 0
            time.sleep(a.interval)
            continue
        events = []
        for name, st in sorted(states.items()):
            old = wm["states"].get(name)
            if old is not None and old != st:
                events.append(f"JOB {name} {old} -> {st}")
            wm["states"][name] = st
        frac = min(1.0, max(q / TARGET, f / CANDIDATES))
        mark = int((5 + 55 * frac) // 5) * 5 if (q or f) else 0
        if mark > wm["mark"]:
            events.append(f"PROGRESS {mark}% overall | hard screening: {q}/{TARGET} qualified, "
                          f"{f}/{CANDIDATES} candidates finished")
            wm["mark"] = mark
        for e in errors:
            if e not in wm["errors"]:
                events.append(f"LOGERR {e[:220]}")
                wm["errors"].append(e)
        wm_path.write_text(json.dumps(wm))
        if events:
            print("\n".join(events), flush=True)
            return 0
        time.sleep(a.interval)
    print("WATCH max-hours reached without events", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
