from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


OPEN_MIDI_BY_STRING = {0: 69, 1: 64, 2: 60, 3: 67}


def extract_difficulty_map(data: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in data:
        song_id = item.get("song_id")
        exercise_id = item.get("exercise_id")
        difficulty_level = item.get("difficulty_level")

        if song_id is None or exercise_id is None or difficulty_level is None:
            continue

        key = f"{song_id}^{exercise_id}"
        result[key] = difficulty_level
    return result


def extract_difficulty_map_from_song_info_map(song_info_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, entry in song_info_map.items():
        difficulty_level = entry.get("difficulty_level")
        if difficulty_level is None:
            continue
        result[key] = difficulty_level
    return result


def extract_song_info(data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for item in data:
        song_id = item.get("song_id")
        exercise_id = item.get("exercise_id")
        difficulty_level = item.get("difficulty_level")
        events_data_raw = item.get("events_data")

        if song_id is None or exercise_id is None:
            continue

        key = f"{song_id}_{exercise_id}"
        pitches: list[Any] = []
        duration: list[Any] = []
        strings: list[Any] = []

        if events_data_raw:
            try:
                events_data = json.loads(events_data_raw)
                inner_data = events_data.get("data", {})
                pitches = inner_data.get("pitches") or []
                duration = inner_data.get("duration") or []
                strings = inner_data.get("strings") or []
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: failed to parse events_data for key={key}")
                continue

        entry = {
            "difficulty_level": difficulty_level,
            "pitches": pitches,
            "duration": duration,
            "strings": strings,
        }

        new_pitch_len = len(pitches)
        if key not in result:
            result[key] = entry
        else:
            old_pitch_len = len(result[key].get("duration") or [])
            if new_pitch_len > old_pitch_len:
                result[key] = entry

    return result


def _to_int_list(x: Any) -> List[int]:
    if isinstance(x, list):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                continue
        return out

    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.replace(",", "^")
    if not s:
        return []
    out = []
    for t in s.split("^"):
        t = t.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except (TypeError, ValueError):
            try:
                out.append(int(float(t)))
            except (TypeError, ValueError):
                continue
    return out


def _cumulative_pitch_list(x: Any) -> List[int]:
    vals = _to_int_list(x)
    if not vals:
        return []
    out = [vals[0]]
    for d in vals[1:]:
        out.append(out[-1] + d)
    return out


def _fret4_list_from_pitch_and_string(pitch_x: Any, string_x: Any) -> List[int]:
    mids = _cumulative_pitch_list(pitch_x)
    strs = _to_int_list(string_x)
    if len(mids) != len(strs):
        raise ValueError(f"mismatch lengths: mids={mids!r} vs strings={strs!r}")
    fret4 = [0, 0, 0, 0]
    for m, s in zip(mids, strs):
        if s not in OPEN_MIDI_BY_STRING:
            raise ValueError(f"invalid string index: {s}")
        fret4[s] = int(m) - OPEN_MIDI_BY_STRING[s]
    return fret4


def convert_song_info_map_to_fret_strings(
    song_info_map: Dict[str, Dict[str, Any]]
) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    converted: Dict[str, Dict[str, Any]] = {}
    skipped_entries: List[Dict[str, Any]] = []
    skipped = 0

    for key, entry in song_info_map.items():
        pitches = entry.get("pitches") or []
        strings = entry.get("strings") or []
        duration = entry.get("duration") or []
        difficulty_level = entry.get("difficulty_level")

        n = min(len(pitches), len(strings), len(duration) if duration else len(pitches))
        if n == 0:
            skipped += 1
            skipped_entries.append(
                {
                    "key": key,
                    "reason": "empty_or_length_zero",
                    "pitches_len": len(pitches),
                    "strings_len": len(strings),
                    "duration_len": len(duration),
                }
            )
            continue

        fret_seq: List[List[int]] = []
        string_seq: List[List[int]] = []
        duration_seq: List[Any] = []

        for i in range(n):
            try:
                fret4 = _fret4_list_from_pitch_and_string(pitches[i], strings[i])
                strs_sorted = sorted(set(_to_int_list(strings[i])))
                if not strs_sorted:
                    skipped_entries.append(
                        {
                            "key": key,
                            "index": i,
                            "reason": "empty_strings_after_parse",
                            "raw_strings": strings[i],
                        }
                    )
                    continue
                fret_seq.append(fret4)
                string_seq.append(strs_sorted)
                if duration:
                    duration_seq.append(duration[i])
            except ValueError as e:
                skipped_entries.append(
                    {
                        "key": key,
                        "index": i,
                        "reason": "invalid_pitch_string_pair",
                        "error": str(e),
                        "raw_pitch": pitches[i],
                        "raw_string": strings[i],
                    }
                )
                continue

        if not fret_seq:
            skipped += 1
            skipped_entries.append(
                {
                    "key": key,
                    "reason": "no_valid_frames_after_filtering",
                    "pitches_len": len(pitches),
                    "strings_len": len(strings),
                    "duration_len": len(duration),
                }
            )
            continue

        new_entry: Dict[str, Any] = {
            "difficulty_level": difficulty_level,
            "fret": fret_seq,
            "strings": string_seq,
        }
        if duration_seq:
            new_entry["duration"] = duration_seq

        converted[key] = new_entry

    print(f"Converted song_info_map entries: {len(converted)}, skipped: {skipped}")
    return converted, skipped_entries


def load_json(json_path: Path) -> Any:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build song_info_map.json, song_info_fret_strings_map.json and "
            "difficulty_map.json from Yousician JSON."
        )
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to source JSON file (e.g. yousician_ukulele_original.json).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output files (default: current directory).",
    )
    parser.add_argument(
        "--song-info-name",
        default="song_info_map.json",
        help="Output filename for song info map.",
    )
    parser.add_argument(
        "--song-info-fret-name",
        default="song_info_fret_strings_map.json",
        help="Output filename for song info fret+strings map.",
    )
    parser.add_argument(
        "--difficulty-name",
        default="difficulty_map.json",
        help="Output filename for difficulty map.",
    )
    parser.add_argument(
        "--skip-report",
        default="",
        help="Optional filename for skipped entries report (JSON).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)

    data = load_json(input_json)
    if isinstance(data, list):
        song_info_map = extract_song_info(data)
        song_info_fret_map, skipped_entries = convert_song_info_map_to_fret_strings(song_info_map)
        difficulty_map = extract_difficulty_map(data)
        print("Detected input format: raw list JSON.")
    elif isinstance(data, dict):
        song_info_map = data
        song_info_fret_map, skipped_entries = convert_song_info_map_to_fret_strings(song_info_map)
        difficulty_map = extract_difficulty_map_from_song_info_map(song_info_map)
        print("Detected input format: song_info_map JSON.")
    else:
        raise ValueError("Unsupported input JSON format. Must be list or dict.")

    song_info_path = output_dir / args.song_info_name
    song_info_fret_path = output_dir / args.song_info_fret_name
    difficulty_path = output_dir / args.difficulty_name

    save_json(song_info_map, song_info_path)
    save_json(song_info_fret_map, song_info_fret_path)
    save_json(difficulty_map, difficulty_path)
    if args.skip_report:
        skip_report_path = output_dir / args.skip_report
        save_json({"skipped_entries": skipped_entries}, skip_report_path)
        print(f"Done. skip_report: {skip_report_path}")
        print(f"skip_report size: {len(skipped_entries)}")

    print(f"Done. song_info_map: {song_info_path}")
    print(f"Done. song_info_fret_map: {song_info_fret_path}")
    print(f"Done. difficulty_map: {difficulty_path}")
    print(f"song_info_map size: {len(song_info_map)}")
    print(f"song_info_fret_map size: {len(song_info_fret_map)}")
    print(f"difficulty_map size: {len(difficulty_map)}")


if __name__ == "__main__":
    main()
