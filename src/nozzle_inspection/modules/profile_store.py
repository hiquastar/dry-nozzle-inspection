"""
Manual-profile persistence helpers.
"""

import json
import os


def load_manual_zone_payload(path):
    """Read and parse the JSON profile file at path, returning the dict payload or None on failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def save_manual_zone_payload(path, payload):
    """Write the dict payload to path as indented JSON, creating parent directories as needed."""
    if not isinstance(payload, dict):
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def extract_zones_from_payload(payload, profile_name):
    """Return the zone list for the named profile, falling back to 'default' then legacy 'zones'."""
    if not isinstance(payload, dict):
        return []
    profiles = payload.get("profiles")
    if isinstance(profiles, dict):
        chosen_name = (
            str(profile_name or payload.get("active_profile", "default"))
            .strip()
            .lower()
        )
        if not chosen_name:
            chosen_name = "default"
        zone_list = profiles.get(chosen_name)
        if zone_list is None and chosen_name != "default":
            zone_list = profiles.get("default")
        if isinstance(zone_list, list):
            return zone_list
        return []
    zone_list = payload.get("zones")
    return zone_list if isinstance(zone_list, list) else []


def extract_profiles_from_payload(payload):
    """Return a normalized {lowercased_name: zone_list} map, wrapping legacy zones under 'default'."""
    if not isinstance(payload, dict):
        return {"default": []}
    profiles = payload.get("profiles")
    if isinstance(profiles, dict):
        clean = {}
        for name, zones in profiles.items():
            key = str(name).strip().lower()
            if not key:
                continue
            clean[key] = zones if isinstance(zones, list) else []
        if clean:
            return clean
    legacy = payload.get("zones") if isinstance(payload.get("zones"), list) else []
    return {"default": legacy}


def persist_manual_zone_active_profile(path, current_profile):
    """Write the chosen profile as active_profile (validated against existing) and return its name."""
    payload = load_manual_zone_payload(path)
    if not isinstance(payload, dict):
        payload = {}
    profiles = extract_profiles_from_payload(payload)
    name = str(current_profile or "default").strip().lower()
    if not name:
        name = "default"
    if name not in profiles:
        name = sorted(profiles.keys())[0]
    payload = {
        "active_profile": name,
        "profiles": profiles,
    }
    save_manual_zone_payload(path, payload)
    return name


def list_manual_zone_profiles(path):
    """Return the sorted list of profile names in the file, or ['default'] when none exist."""
    payload = load_manual_zone_payload(path)
    if not isinstance(payload, dict):
        return ["default"]
    profiles = extract_profiles_from_payload(payload)
    names = sorted(
        [str(name).strip().lower() for name in profiles.keys() if str(name).strip()]
    )
    return names if names else ["default"]


def delete_manual_zone_profile(path, profile_name):
    """Remove the named profile, persist the result, and return (success, next_active_name)."""
    target = str(profile_name or "").strip().lower()
    if not target:
        return False, None
    payload = load_manual_zone_payload(path)
    if not isinstance(payload, dict):
        return False, None
    profiles = extract_profiles_from_payload(payload)
    if target not in profiles:
        return False, None
    del profiles[target]
    if not profiles:
        profiles = {"default": []}
    next_name = sorted(profiles.keys())[0]
    payload = {
        "active_profile": next_name,
        "profiles": profiles,
    }
    save_manual_zone_payload(path, payload)
    return True, next_name


def suggest_new_manual_profile_name(path, base="profile"):
    """Return the first '{base}_N' name not already used by an existing profile in the file."""
    payload = load_manual_zone_payload(path)
    profiles = extract_profiles_from_payload(
        payload if isinstance(payload, dict) else {}
    )
    existing = {
        str(name).strip().lower() for name in profiles.keys() if str(name).strip()
    }
    idx = 1
    candidate = f"{base}_{idx}"
    while candidate in existing:
        idx += 1
        candidate = f"{base}_{idx}"
    return candidate
