#!/usr/bin/env bash
# Shared powermetrics energy capture for the benchmark harness.
#
# Source this, then set BENCH, MODEL and RUN_DIR before calling:
#   energy_detect              -> sets PM_BIN and HAVE_ENERGY (0/1)
#   energy_run OUT --args...   -> runs $BENCH wrapped in powermetrics and
#                                 merges power/residency/joules into OUT
#
# powermetrics needs root; energy_detect checks passwordless sudo and prints a
# one-line grant command if missing.

energy_detect() {
  PM_BIN="$(command -v powermetrics || true)"
  HAVE_ENERGY=0
  if [ -z "$PM_BIN" ]; then
    echo "energy: powermetrics not found; skipping" >&2
    return
  fi
  if sudo -n "$PM_BIN" -n 1 -i 50 -s gpu_power -o /dev/null >/dev/null 2>&1; then
    HAVE_ENERGY=1
  elif [ -t 0 ]; then
    echo "energy: sudo needed for powermetrics (one prompt)..."
    if sudo -v 2>/dev/null && sudo -n "$PM_BIN" -n 1 -i 50 -s gpu_power -o /dev/null >/dev/null 2>&1; then
      HAVE_ENERGY=1
    else
      echo "energy: sudo failed; skipping" >&2
    fi
  else
    echo "energy: no passwordless sudo for powermetrics; skipping" >&2
    echo "  grant it once with:" >&2
    echo "    echo \"$(id -un) ALL=(root) NOPASSWD: $PM_BIN\" | sudo tee /etc/sudoers.d/powermetrics >/dev/null && sudo chmod 440 /etc/sudoers.d/powermetrics" >&2
  fi
}

energy_run() {
  # $1 = output json; rest = bench args. Wraps the run in powermetrics and
  # merges avg power / residency / joules into the JSON. Always reaps the
  # privileged sampler, even if the benchmark fails.
  local out="$1"; shift
  local pmf; pmf="$(mktemp)"
  local pmlog="$RUN_DIR/$(basename "$out" .json).powermetrics.txt"
  sudo powermetrics --samplers cpu_power,gpu_power,thermal,ane_power --show-process-gpu \
       -i 250 -o "$pmf" >/dev/null 2>&1 &
  local pm_pid=$!
  sleep 1
  local rc=0
  # shellcheck disable=SC2086
  $BENCH --model "$MODEL" --seed 42 "$@" --out "$out" >/dev/null || rc=$?
  kill "$pm_pid" 2>/dev/null || true
  wait "$pm_pid" 2>/dev/null || true
  python3 - "$out" "$pmf" <<'PY' || true
import json, re, sys, datetime, os
out_path, pm_path = sys.argv[1], sys.argv[2]
d = json.load(open(out_path))
r, c = d["results"], d["config"]
t0 = r.get("active_started_unix", 0)
t1 = r.get("active_ended_unix", 0)

def parse_ts(s):
    try:
        return datetime.datetime.strptime(s.strip(), "%a %b %d %H:%M:%S %Y %z").timestamp()
    except Exception:
        return None

# Optional metrics: (json key, regex, unit scale). Absent fields are omitted, so
# the parser stays valid across macOS/powermetrics versions.
PATTERNS = [
    ("avg_cpu_w",       r'CPU Power:\s*([\d.]+)\s*mW', 0.001),
    ("avg_gpu_w",       r'GPU Power:\s*([\d.]+)\s*mW', 0.001),
    ("avg_ane_w",       r'ANE Power:\s*([\d.]+)\s*mW', 0.001),
    ("combined_w",      r'Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW', 0.001),
    ("gpu_active_pct",  r'GPU HW [Aa]ctive [Rr]esidency:\s*([\d.]+)\s*%', 1.0),
    ("gpu_freq_mhz",    r'GPU HW [Aa]ctive [Ff]requency:\s*([\d.]+)\s*MHz', 1.0),
    ("p_cluster_w",     r'P\d*-Cluster Power:\s*([\d.]+)\s*mW', 0.001),
    ("e_cluster_w",     r'E\d*-Cluster Power:\s*([\d.]+)\s*mW', 0.001),
    ("p_residency_pct", r'P\d*-Cluster HW [Aa]ctive [Rr]esidency:\s*([\d.]+)\s*%', 1.0),
    ("e_residency_pct", r'E\d*-Cluster HW [Aa]ctive [Rr]esidency:\s*([\d.]+)\s*%', 1.0),
    ("p_freq_mhz",      r'P\d*-Cluster HW [Aa]ctive [Ff]requency:\s*([\d.]+)\s*MHz', 1.0),
    ("e_freq_mhz",      r'E\d*-Cluster HW [Aa]ctive [Ff]requency:\s*([\d.]+)\s*MHz', 1.0),
    ("cpu_die_c",       r'CPU die temperature:\s*([\d.]+)\s*C', 1.0),
    ("gpu_die_c",       r'GPU die temperature:\s*([\d.]+)\s*C', 1.0),
]

def new_block():
    return {"ts": None, "vals": {k: [] for (k, _, _) in PATTERNS}}

# Split the powermetrics stream into per-sample blocks.
blocks = []
cur = new_block()
for line in open(pm_path, errors='ignore'):
    m = re.search(r'\*\*\* Sampled system activity \(([^)]*)\)', line)
    if m:
        if cur["ts"] is not None:
            blocks.append(cur)
        cur = new_block(); cur["ts"] = parse_ts(m.group(1))
        continue
    if cur["ts"] is None:
        continue
    if "Average" in line or "(avg" in line:      # skip powermetrics' own averages
        continue
    for (key, rx, scale) in PATTERNS:
        mm = re.search(rx, line)
        if mm:
            cur["vals"][key].append(float(mm.group(1)) * scale)
if cur["ts"] is not None:
    blocks.append(cur)

# Only samples inside the harness's measured window (small tolerance);
# fall back to all samples if the window is unavailable.
sel = [b for b in blocks if t0 and t1 and (t0 - 0.5) <= b["ts"] <= (t1 + 0.5)]
if not sel:
    sel = blocks

def mean(key):
    xs = [v for b in sel for v in b["vals"][key]]
    return (sum(xs) / len(xs)) if xs else 0.0

energy = {"samples_used": len(sel)}
for (key, _, _) in PATTERNS:
    xs = [v for b in sel for v in b["vals"][key]]
    if xs:
        energy[key] = round(sum(xs) / len(xs), 2)

# Total power: measured SoC rails (CPU+GPU+ANE); fall back to the combined line.
cpu_w, gpu_w, ane_w = mean("avg_cpu_w"), mean("avg_gpu_w"), mean("avg_ane_w")
total_w = cpu_w + gpu_w + ane_w
if total_w == 0.0:
    total_w = mean("combined_w")
energy["avg_cpu_w"] = round(cpu_w, 2)
energy["avg_gpu_w"] = round(gpu_w, 2)
energy["avg_ane_w"] = round(ane_w, 2)
energy["avg_w"] = round(total_w, 2)
# raw powermetrics (incl. per-process GPU via --show-process-gpu) kept for
# inspection; per-process extraction is left to the reader for now.
energy["powermetrics_log"] = os.path.basename(out_path).replace(".json", ".powermetrics.txt")

active = r.get("active_seconds", 0.0)
reps = c.get("reps", 1)
joules = total_w * active
energy["joules"] = round(joules, 2)
if c.get("gen_tokens", 0) > 8:
    dec_tokens = (c["gen_tokens"] - 1) * reps
    energy["j_per_decode_token"] = round(joules / dec_tokens, 4) if dec_tokens else 0.0
if c.get("prompt_tokens", 0) > 128 and c.get("gen_tokens", 0) <= 8:
    pre_tokens = c["prompt_tokens"] * reps
    energy["j_per_prefill_token"] = round(joules / pre_tokens, 4) if pre_tokens else 0.0
gen = r.get("generated_tokens", 0)
energy["j_per_token"] = round(joules / gen, 4) if gen else 0.0
r["energy"] = energy
json.dump(d, open(out_path, "w"), indent=2)
extra = []
for k in ("cpu_die_c", "gpu_die_c", "gpu_freq_mhz", "gpu_active_pct"):
    if k in energy:
        extra.append(f"{k}={energy[k]}")
print(f"  {out_path}: {total_w:.2f} W (cpu {cpu_w:.2f} + gpu {gpu_w:.2f} + ane {ane_w:.2f}), "
      f"{joules:.1f} J, {len(sel)} samples" + (", " + ", ".join(extra) if extra else ""))
PY
  cp -f "$pmf" "$pmlog" 2>/dev/null || true
  rm -f "$pmf"
  return "$rc"
}
