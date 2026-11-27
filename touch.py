"""Palmimo can feel: touch reactions from leg-servo current (no extra sensors).

Probe results (probe.py, 2026-11-27): ~126 Hz sweeps, idle noise sd ~1-2 raw.
  * grip front-right leg: leg_6_pitch1 jumps ~+150  -> HANDSHAKE (very reliable)
  * poke front-right leg: leg_6_pitch2 transient ~28 -> POKE
  * press body: spread shift ~25-35 on several legs  -> PRESS (tune --press)
Absolute levels drift after any disturbance (stiction), so every signal is the
deviation from a slow EMA baseline that is frozen while touched and re-seeded
after every reaction.

Logs: stdout (Portal) + logs/touch_<stamp>.log. A status line every
--status seconds shows the live deviations so thresholds can be tuned.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from palmimo_sdk import DynamixelDriver, FaceDisplay, Palmimo

LEGS = [f"leg_{i}_{j}" for i in range(1, 7) for j in ("yaw", "pitch1", "pitch2")]
HAND = ["leg_6_pitch1", "leg_6_pitch2"]  # front-right leg (the one that was gripped)
# Axes that hunted around zero in the probe (sd 4-9) - excluded from PRESS sum.
NOISY = {"leg_4_yaw", "leg_5_yaw", "leg_5_pitch1"}
BODY = [m for m in LEGS if m not in NOISY and not m.startswith("leg_6")]

log = logging.getLogger("touch")


def setup_logging(stamp: str) -> None:
    Path("logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(f"logs/touch_{stamp}.log", encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)


class Touch:
    def __init__(self, robot: Palmimo, a: argparse.Namespace):
        self.r, self.a = robot, a
        self.ema: dict[str, float] = {}
        self.over_since: dict[str, float] = {}

    # ---- sensing -------------------------------------------------------
    def read(self) -> dict[str, int] | None:
        try:
            tel = self.r.driver.read_telemetry(LEGS)
        except Exception as e:
            log.warning("telemetry read failed: %r", e)
            return None
        if tel.silent or tel.unreached:
            log.warning("incomplete sweep silent=%s unreached=%s", tel.silent, tel.unreached)
        return dict(tel.current)

    def rebaseline(self, settle: float) -> None:
        time.sleep(settle)
        acc: dict[str, list[int]] = {}
        end = time.monotonic() + 0.4
        while time.monotonic() < end:
            cur = self.read() or {}
            for m, c in cur.items():
                acc.setdefault(m, []).append(c)
        self.ema = {m: sum(v) / len(v) for m, v in acc.items()}
        self.over_since.clear()
        log.info("baseline: hand=%s", {m: round(self.ema.get(m, 0)) for m in HAND})

    def dev(self, cur: dict[str, int]) -> dict[str, float]:
        return {m: c - self.ema.get(m, c) for m, c in cur.items()}

    def track(self, cur: dict[str, int], d: dict[str, float], dt: float) -> None:
        """Slow EMA, frozen per-axis while that axis is clearly disturbed."""
        alpha = min(1.0, dt / self.a.tau)
        for m, c in cur.items():
            if abs(d[m]) < self.a.poke:
                self.ema[m] = self.ema.get(m, c) + alpha * (c - self.ema.get(m, c))

    def held(self, key: str, cond: bool, now: float, hold: float) -> bool:
        if not cond:
            self.over_since.pop(key, None)
            return False
        t0 = self.over_since.setdefault(key, now)
        return now - t0 >= hold

    # ---- acting --------------------------------------------------------
    def face(self, name: str, hold_ms: int = 0) -> None:
        if self.r.display is None:
            return
        try:
            self.r.set_expression(name, hold_ms=hold_ms)
        except Exception as e:
            log.warning("expression %s failed: %r", name, e)

    def motion(self, name: str, seconds: float) -> None:
        log.info("  -> motion %s (%.1fs)", name, seconds)
        try:
            self.r.set_motion(name)
            self.r.run(seconds=seconds)
            self.r.set_motion("idle")
            self.r.run(seconds=1.0)
        except Exception as e:
            log.error("motion %s failed: %r", name, e)
        trips = self.r.drain_overload_trips()
        if trips:
            log.warning("  overload trips: %s", trips)

    def on_grip(self) -> None:
        log.info("EVENT HANDSHAKE")
        self.face("LOVE")
        # Hold still while they shake; wait for release (or timeout).
        end = time.monotonic() + self.a.grip_max
        while time.monotonic() < end:
            cur = self.read()
            if cur and abs(cur["leg_6_pitch1"] - self.ema["leg_6_pitch1"]) < self.a.grip / 2:
                log.info("  released")
                break
        self.face("HAPPY")
        self.motion("nod", 2.0)
        self.motion("wave", 3.0)

    def on_poke(self, peak: float) -> None:
        log.info("EVENT POKE (peak dev %.0f)", peak)
        self.face("SURPRISED")
        self.motion("head_shake", 2.0)

    def on_press(self, total: float) -> None:
        log.info("EVENT PRESS (sum dev %.0f)", total)
        self.face("ANGRY")
        self.motion("pushup", 2.5)

    # ---- loop ----------------------------------------------------------
    def loop(self) -> None:
        self.face("HAPPY")
        self.rebaseline(1.0)
        log.info("READY: shake front-right leg / poke it / press its back (if --press > 0)")
        last, last_status = time.monotonic(), 0.0
        poke_until: float | None = None
        poke_peak = 0.0
        while True:
            cur = self.read()
            now = time.monotonic()
            dt, last = now - last, now
            if cur is None:
                time.sleep(0.02)
                continue
            d = self.dev(cur)
            hand1 = abs(d.get("leg_6_pitch1", 0))
            hand_max = max(abs(d.get(m, 0)) for m in HAND)
            body = sum(abs(d.get(m, 0)) for m in BODY)

            if now - last_status >= self.a.status:
                last_status = now
                log.info("  live: d6p1=%+5.0f d6p2=%+5.0f body_sum=%4.0f",
                         d.get("leg_6_pitch1", 0), d.get("leg_6_pitch2", 0), body)

            fired = None
            if self.held("grip", hand1 > self.a.grip, now, self.a.grip_hold):
                fired = self.on_grip
            elif self.a.press > 0 and self.held("press", body > self.a.press, now, 0.4):
                fired = lambda: self.on_press(body)  # noqa: E731
            else:
                # Poke = short spike on the hand that does NOT become a grip.
                if hand_max > self.a.poke and poke_until is None:
                    poke_until, poke_peak = now + self.a.grip_hold + 0.1, hand_max
                if poke_until is not None:
                    poke_peak = max(poke_peak, hand_max)
                    if now >= poke_until:
                        poke_until = None
                        if hand1 <= self.a.grip:
                            pk = poke_peak
                            fired = lambda: self.on_poke(pk)  # noqa: E731

            if fired:
                poke_until = None
                fired()
                self.rebaseline(self.a.settle)
                log.info("READY")
                continue
            self.track(cur, d, dt)


def main() -> int:
    p = argparse.ArgumentParser(description="Palmimo touch reactions")
    p.add_argument("--grip", type=float, default=60, help="hand dev threshold for handshake (probe: ~150)")
    p.add_argument("--grip-hold", type=float, default=0.25, help="s above --grip to count as handshake")
    p.add_argument("--grip-max", type=float, default=4.0, help="max s to wait for release")
    p.add_argument("--poke", type=float, default=15, help="hand dev threshold for poke (probe: ~28)")
    p.add_argument("--press", type=float, default=0, help="body sum threshold for press; 0 = off (try 120)")
    p.add_argument("--tau", type=float, default=3.0, help="baseline EMA time constant, s")
    p.add_argument("--settle", type=float, default=1.0, help="s to settle after a reaction")
    p.add_argument("--status", type=float, default=0.5, help="s between live status lines")
    p.add_argument("--no-face", action="store_true")
    a = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(stamp)
    log.info("args: %s", vars(a))

    robot = None
    if not a.no_face:
        try:
            robot = Palmimo(driver=DynamixelDriver(), display=FaceDisplay())
            robot.connect()
        except Exception as e:
            log.warning("face display unavailable (%r) - continuing without it", e)
            try:
                robot.disconnect()
            except Exception:
                pass
            robot = None
    if robot is None:
        robot = Palmimo(driver=DynamixelDriver())
        robot.connect()
    log.info("connected (display=%s)", robot.display is not None)
    try:
        Touch(robot, a).loop()
    finally:
        log.info("disconnecting")
        robot.disconnect()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warning("interrupted")
        sys.exit(130)
    except Exception:
        log.error("FAILED:\n%s", traceback.format_exc())
        sys.exit(1)
