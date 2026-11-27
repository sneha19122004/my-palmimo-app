"""Feasibility probe: can the leg servos' current readings detect touch / load?

Timed phases (no keyboard needed, so it runs from the Portal too). Watch the
log and do what each phase says while it counts down:

  baseline -> poke_leg -> grip_leg -> press_back -> weight -> rest

Outputs:
  * stdout (Portal "Logs"): phase cues, a 1 Hz live line, per-phase summary,
    and a final GO / NO-GO verdict.
  * logs/probe_<timestamp>.csv: every sweep, every leg axis
    (t, phase, motor, current, position, temperature, voltage).
  * logs/probe_<timestamp>.log: copy of stdout.

The robot only holds its standing pose: nothing moves after the wake glide.
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from palmimo_sdk import DynamixelDriver, Palmimo

LEGS = [f"leg_{i}_{j}" for i in range(1, 7) for j in ("yaw", "pitch1", "pitch2")]
LEG_OVERLOAD = 1200  # SDK default trip threshold (raw units) - stay well below

log = logging.getLogger("probe")


def setup_logging(log_dir: Path, stamp: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"probe_{stamp}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    return log_path


def stats(vals: list[int]) -> tuple[float, float, int, int]:
    if not vals:
        return (float("nan"), float("nan"), 0, 0)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return (statistics.fmean(vals), sd, min(vals), max(vals))


def main() -> int:
    p = argparse.ArgumentParser(description="Palmimo servo-current touch probe")
    p.add_argument("--baseline", type=float, default=8.0, help="seconds untouched")
    p.add_argument("--phase", type=float, default=10.0, help="seconds per touch phase")
    p.add_argument("--countdown", type=float, default=4.0, help="seconds of 'get ready' before each phase")
    p.add_argument("--port", default=None, help="servo port (default: auto-detect)")
    p.add_argument("--log-dir", default="logs")
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)
    log_path = setup_logging(log_dir, stamp)
    csv_path = log_dir / f"probe_{stamp}.csv"
    log.info("palmimo-sdk %s | log %s | csv %s", version("palmimo-sdk"), log_path, csv_path)

    phases = [
        ("baseline", args.baseline, "HANDS OFF. Do not touch the robot or the table."),
        ("poke_leg", args.phase, "POKE the front-right leg (leg 6) a few times, gently, ~1 s apart."),
        ("grip_leg", args.phase, "GRIP and lightly push/pull the front-right leg (handshake), then let go."),
        ("press_back", args.phase, "PRESS DOWN on the top of the body with a flat hand - light, then firmer. Not hard."),
        ("weight", args.phase, "PLACE an object (phone ~200 g) centred on its back. Leave it there. Remove at end."),
        ("rest", args.baseline, "HANDS OFF again (checks drift back to baseline)."),
    ]

    robot = Palmimo(driver=DynamixelDriver(port=args.port))
    log.info("connecting (robot will do its wake glide)...")
    with robot, open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        drv = robot.driver
        w = csv.writer(fcsv)
        w.writerow(["t", "phase", "motor", "current", "position", "temperature", "voltage"])

        # Health check before we start (works on palmimo-sdk 0.1.4).
        try:
            tel = drv.read_telemetry()
            log.info("startup: Vmin=%.2f Tmax=%sC silent=%s unreached=%s",
                     min(tel.voltage.values(), default=float("nan")),
                     max(tel.temperature.values(), default="?"), tel.silent, tel.unreached)
        except Exception as e:
            log.warning("startup telemetry failed: %r", e)

        time.sleep(1.0)  # let the stance settle after wake
        cur_by_phase: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        sweeps = defaultdict(int)
        sweep_ms: list[float] = []
        misses = 0
        t0 = time.monotonic()

        for name, dur, cue in phases:
            log.info("=" * 70)
            log.info("NEXT PHASE: %s (%.0fs) -> %s", name.upper(), dur, cue)
            for k in range(int(args.countdown), 0, -1):
                log.info("  starting in %d...", k)
                time.sleep(1.0)
            log.info(">>> GO: %s", name.upper())
            end = time.monotonic() + dur
            last_print = 0.0
            while (now := time.monotonic()) < end:
                ts = time.perf_counter()
                try:
                    tel = drv.read_telemetry(LEGS)
                except Exception as e:
                    misses += 1
                    log.warning("telemetry read failed: %r", e)
                    time.sleep(0.05)
                    continue
                sweep_ms.append((time.perf_counter() - ts) * 1000)
                if tel.silent or tel.unreached:
                    misses += 1
                    log.warning("sweep incomplete silent=%s unreached=%s", tel.silent, tel.unreached)
                t = round(now - t0, 4)
                sweeps[name] += 1
                for m, c in tel.current.items():
                    cur_by_phase[name][m].append(c)
                    w.writerow([t, name, m, c, tel.position.get(m), tel.temperature.get(m), tel.voltage.get(m)])
                    if abs(c) >= 0.8 * LEG_OVERLOAD:
                        log.warning("!! %s current %d near overload (%d) - ease off", m, c, LEG_OVERLOAD)
                if now - last_print >= 1.0:
                    last_print = now
                    total = sum(abs(c) for c in tel.current.values())
                    top = sorted(tel.current.items(), key=lambda kv: -abs(kv[1]))[:3]
                    vmin = min(tel.voltage.values(), default=float("nan"))
                    tmax = max(tel.temperature.values(), default=0)
                    log.info("  [%s %4.1fs left] sum|I|=%5d top=%s  Vmin=%.2f Tmax=%dC",
                             name, end - now, total, top, vmin, tmax)
            fcsv.flush()

        # ---------- analysis ----------
        log.info("=" * 70)
        n_sweeps = sum(sweeps.values())
        elapsed = time.monotonic() - t0
        if sweep_ms:
            log.info("SWEEP RATE: %d sweeps, mean %.1f ms/sweep (max %.1f) => ~%.0f Hz possible; misses=%d",
                     n_sweeps, statistics.fmean(sweep_ms), max(sweep_ms),
                     1000 / statistics.fmean(sweep_ms), misses)
        log.info("elapsed %.1fs", elapsed)

        base = cur_by_phase["baseline"]
        base_stats = {m: stats(v) for m, v in base.items()}
        log.info("BASELINE noise per axis (mean / sd / min..max):")
        for m in LEGS:
            mu, sd, lo, hi = base_stats.get(m, (float("nan"),) * 4)
            log.info("  %-14s %7.1f / %5.1f / %5s..%-5s", m, mu, sd, lo, hi)

        verdict_lines = []
        for name, _, _ in phases[1:]:
            log.info("PHASE %s: biggest deviations from baseline (|peak - base mean| and in baseline-SDs):", name)
            rows = []
            for m, vals in cur_by_phase[name].items():
                mu, sd, _, _ = base_stats.get(m, (0.0, 0.0, 0, 0))
                peak = max(vals, key=lambda c: abs(c - mu))
                dev = abs(peak - mu)
                shift = abs(statistics.fmean(vals) - mu)
                snr = dev / max(sd, 1.0)
                rows.append((snr, m, dev, shift, peak))
            rows.sort(reverse=True)
            for snr, m, dev, shift, peak in rows[:5]:
                log.info("  %-14s peak=%6d dev=%6.0f meanShift=%6.0f SNR=%5.1f", m, peak, dev, shift, snr)
            best = rows[0] if rows else (0, "-", 0, 0, 0)
            verdict_lines.append((name, best[0], best[1]))

        log.info("=" * 70)
        log.info("VERDICT (SNR = peak deviation / baseline sd; want > ~5 for a reliable trigger):")
        for name, snr, m in verdict_lines:
            tag = "GO" if snr >= 5 else ("MAYBE" if snr >= 3 else "NO-GO")
            log.info("  %-11s best axis %-14s SNR %5.1f  -> %s", name, m, snr, tag)
        rest_shift = max((abs(statistics.fmean(v) - base_stats[m][0])
                          for m, v in cur_by_phase["rest"].items() if m in base_stats), default=0)
        log.info("  drift after rest: max mean shift %.0f raw units (large => need re-baselining)", rest_shift)
        rail = getattr(robot, "drain_rail_trips", lambda: "n/a in this SDK")()
        log.info("overload trips: %s | rail trips: %s", robot.drain_overload_trips() or "none", rail or "none")
        log.info("done. CSV: %s", csv_path.resolve())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warning("interrupted - robot disconnected/parked by context manager")
        sys.exit(130)
    except Exception:
        log.error("FAILED:\n%s", traceback.format_exc())
        sys.exit(1)
