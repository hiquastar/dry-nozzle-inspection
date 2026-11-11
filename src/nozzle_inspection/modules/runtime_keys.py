"""
Runtime keyboard/mode handlers extracted from run_inspection.py.
"""

import time

import numpy as np

from . import profile_setting, runtime_io


def reset_runtime_trackers(system):
    """Clear warnings, histories, and per-zone confirmation/reference state for all unlocked zones."""
    print("\nResetting runtime trackers (unlocked zones only)...")
    system.nozzle_detector.reset_reference()
    if system.zone_orchestrator is not None:
        system.zone_orchestrator.reset()
    for history in system.tip_track_history.values():
        history.clear()
    for history in system.zone_confirmation_history.values():
        history.clear()
    for zone_id in range(1, system.total_zones + 1):
        system.zone_confirmed_state[zone_id] = False
        system.zone_ever_confirmed[zone_id] = False
        system.zone_last_confirmed_position[zone_id] = None
        system.zone_last_confirmed_global_position[zone_id] = None
    for zone_id in range(1, system.total_zones + 1):
        system.spray_angle_detector.reset_zone(zone_id)
        if not bool(system.zone_lock_enabled.get(zone_id, False)):
            system.zone_nozzle_detectors[zone_id].reset_reference()
            system._stabilize_debug_bbox(zone_id, None)


def switch_dyn_fix_mode(system):
    """Cycle to the next enabled runtime mode, debounced, exiting MANUAL first if active."""
    if system.current_mode == "MANUAL":
        profile_setting.exit_manual_mode(system)
    enabled_runtime_modes = list(system._enabled_runtime_modes())
    if len(enabled_runtime_modes) <= 1:
        print(
            f"\nZone mode locked: {enabled_runtime_modes[0]} (only enabled runtime mode)"
        )
        return
    now = time.time()
    if (now - float(system.last_mode_toggle_time)) < 0.35:
        return
    system.last_mode_toggle_time = now
    current_mode = str(system.current_mode).upper()
    current_index = (
        enabled_runtime_modes.index(current_mode)
        if current_mode in enabled_runtime_modes
        else 0
    )
    next_mode = enabled_runtime_modes[(current_index + 1) % len(enabled_runtime_modes)]
    system._apply_runtime_mode(next_mode)
    print(f"\nZone mode: {system.current_mode}")


def toggle_profile_setting_mode(system):
    """Enter or exit MANUAL profile-setting mode if enabled, returning True when newly entered."""
    if not bool(system.mode_enabled.get("MANUAL", True)):
        print(
            "\nMANUAL mode is disabled in config (runtime_modes.enable_manual=false)."
        )
        return False
    if system.current_mode == "MANUAL":
        profile_setting.exit_manual_mode(system)
        return False
    entered = profile_setting.enter_manual_mode(system)
    return bool(entered)


def handle_mode_key(system, key):
    """Dispatch the D/F/P/N zone-mode keys, returning True if the key was consumed."""
    if key in (ord("d"), ord("D")):
        if str(system.current_mode).upper() == "VIDEO":
            enabled_runtime_modes = list(system._enabled_runtime_modes())
            preferred = (
                "DYN" if "DYN" in enabled_runtime_modes else enabled_runtime_modes[0]
            )
            system._apply_runtime_mode(preferred)
            print(f"\nZone mode: {system.current_mode}")
            return True
        switch_dyn_fix_mode(system)
        return True
    if key in (ord("f"), ord("F")):
        if not bool(system.mode_enabled.get("FIX", True)):
            print("\nFIX mode is disabled in config.")
            return True
        if system.current_mode == "MANUAL":
            profile_setting.exit_manual_mode(system)
        system._apply_runtime_mode("FIX")
        print(f"\nZone mode: {system.current_mode}")
        return True
    if key in (ord("p"), ord("P")):
        if system.current_mode != "MANUAL":
            print("\nP works only in MANUAL mode.")
            return True
        system._cycle_manual_zone_profile()
        profile_setting.reload_manual_profile_live(system)
        return True
    if key in (ord("n"), ord("N")):
        print("\nN works only in Profile Setting (press M).")
        return True
    return False


def handle_overlay_key(system, key):
    """Toggle the overlay flag mapped to number keys 0-8, returning True if the key was consumed."""
    if key == ord("1"):
        system.show_nozzle_head_circle = not bool(system.show_nozzle_head_circle)
        return True
    if key == ord("0"):
        system.show_nozzle_angle_arrow = not bool(system.show_nozzle_angle_arrow)
        return True
    if key == ord("2"):
        system.show_spray_angle_lines = not bool(system.show_spray_angle_lines)
        return True
    if key == ord("3"):
        system.show_spray_zone_overlay = not bool(system.show_spray_zone_overlay)
        return True
    if key == ord("4"):
        system.show_annular_mask = not bool(system.show_annular_mask)
        return True
    if key == ord("5"):
        system.show_spray_exclusion_mask = not bool(system.show_spray_exclusion_mask)
        return True
    if key == ord("6"):
        system.show_analysis_mask = not bool(system.show_analysis_mask)
        return True
    if key == ord("7"):
        system.show_focus_zone_overlay = not bool(system.show_focus_zone_overlay)
        return True
    if key == ord("8"):
        system.show_focus_zone_panel = not bool(system.show_focus_zone_panel)
        return True
    return False


def handle_lock_key(system, key):
    """Handle the 9/Tab/L zone-lock keys (toggle select mode, cycle zone, lock), returning True if consumed."""
    if key == ord("9"):
        system.lock_zone_select_mode = not bool(system.lock_zone_select_mode)
        system.lock_debug_visible = bool(system.lock_zone_select_mode)
        if system.lock_zone_select_mode:
            system.selected_lock_zone_id = int(
                np.clip(
                    int(system.selected_lock_zone_id), 1, max(1, system.total_zones)
                )
            )
            print(f"\nLock zone select: ON (zone {int(system.selected_lock_zone_id)})")
        else:
            print("\nLock zone select: OFF")
        return True
    if key == 9:
        if system.lock_zone_select_mode:
            system._cycle_selected_zone_lock()
        else:
            system._toggle_buildup_calibration_mode()
        return True
    if key in (ord("l"), ord("L")):
        if system.lock_zone_select_mode:
            system._toggle_zone_lock(zone_id=system.selected_lock_zone_id, quiet=False)
        else:
            print("\nLock-select mode is OFF (press 9 to enable)")
        return True
    return False


def handle_calibration_key(system, key):
    """Adjust annulus offset/band ([];') and spray-exclusion ratio (u/i) on key, returning True if consumed."""
    if key == ord("["):
        system._adjust_manual_annulus_param(offset_delta=-1.0)
        return True
    if key == ord("]"):
        system._adjust_manual_annulus_param(offset_delta=1.0)
        return True
    if key == ord(";"):
        system._adjust_manual_annulus_param(band_delta=-1.0)
        return True
    if key == ord("'"):
        system._adjust_manual_annulus_param(band_delta=1.0)
        return True
    if key in (ord("u"), ord("U")):
        system._adjust_spray_exclusion_param(band_ratio_delta=-1.0)
        return True
    if key in (ord("i"), ord("I")):
        system._adjust_spray_exclusion_param(band_ratio_delta=1.0)
        return True
    return False


def handle_runtime_key(system, key, paused):
    """Route a runtime keypress to the matching handler and return (action, paused) where action is quit/handled/none."""
    if key == ord("q"):
        print("\nShutting down...")
        return "quit", paused
    if key == ord("r"):
        reset_runtime_trackers(system)
        return "handled", paused
    if key in (ord("o"), ord("O")):
        system.video_auto_loop = not bool(getattr(system, "video_auto_loop", True))
        print(f"\nVideo auto-loop: {'ON' if system.video_auto_loop else 'OFF'}")
        return "handled", paused
    if handle_overlay_key(system, key):
        return "handled", paused
    if key in (ord("m"), ord("M")):
        toggled = toggle_profile_setting_mode(system)
        if toggled:
            paused = False
        return "handled", paused
    if key in (ord("i"), ord("I")):
        # I in runtime = force one immediate AI re-classification.
        # (In Profile editor, I has a different meaning: auto-detect new ROIs.)
        ticker = (
            system.get_dl_status_ticker()
            if hasattr(system, "get_dl_status_ticker")
            else None
        )
        if ticker is not None:
            n = ticker.trigger_now()
            print(f"\n[AI] Re-classified {n} zone(s) on demand.")
        else:
            print("\n[AI] DL model not available.")
        return "handled", paused
    if handle_mode_key(system, key):
        return "handled", paused
    if handle_lock_key(system, key):
        return "handled", paused
    if handle_calibration_key(system, key):
        return "handled", paused
    if key in (ord("s"), ord("S")):
        runtime_io.toggle_video_recording(system)
        return "handled", paused
    if key == 32:
        paused = not paused
        print(f"\nPlayback: {'PAUSED' if paused else 'RUNNING'}")
        return "handled", paused
    return "none", paused
