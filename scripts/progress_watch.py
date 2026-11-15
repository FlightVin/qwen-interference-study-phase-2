"""Print one line each time the overall dataset build crosses a 5% mark during hard screening.

Overall progress during hard screening = 5% + 55% x (progress toward whichever stop comes first:
70 qualified problems or the $25 screening budget). Also prints any error line from the log, and
exits when the log shows the screening finished.

Usage: python scripts/progress_watch.py <screening log>
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ri import config  # noqa: E402

STATE = config.SCREENING / "hard_state.json"
log = Path(sys.argv[1])
last_mark = -1
seen_errors = set()
while True:
    try:
        s = json.loads(STATE.read_text())
        q = sum(p["status"] == "qualified" for p in s["problems"].values())
        frac = min(1.0, max(q / config.HARD_TARGET, s["spend_usd"] / config.HARD_BUDGET_USD))
        pct = 5 + 55 * frac
        mark = int(pct // 5) * 5
        if mark > last_mark:
            if last_mark >= 0:
                print(f"PROGRESS {mark}% overall | hard screening: {q} qualified, "
                      f"est. ${s['spend_usd']:.2f} spent, {len(s['problems'])} problems screened",
                      flush=True)
            last_mark = mark
    except Exception:
        pass
    text = log.read_text(errors="ignore") if log.exists() else ""
    for line in text.splitlines():
        if re.search(r"Traceback|Error|metered stop|budget reached|target reached|no candidates", line) \
                and line not in seen_errors:
            seen_errors.add(line)
            print(f"LOG {line.strip()[:200]}", flush=True)
    if re.search(r"^exit \d+", text, re.M):
        print("SCREENING PROCESS EXITED", flush=True)
        break
    time.sleep(60)
