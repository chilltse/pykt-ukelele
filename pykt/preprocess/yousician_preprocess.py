import json
import pandas as pd
from .utils import sta_infos, write_txt

# KC: pitches

# =============================================================================
# yousician JSON 预处理
# -----------------------------------------------------------------------------
# 1) 只保留 play_mode == "play" 的样本
# 2) 按 user_id 分组，同一用户内按 days_since_signup / session / part / record_idx / event_idx 排序后拼接
# 3) question_id = prev_pitches|prev_strings|current_pitches|current_strings
#    prev_* 按 exercise(record) 内计算：
#    - 该 exercise 的起始音为 "inf"
#    - 否则为上一个音的 pitches / strings
# 4) KC(sequence_id) = current_pitches
# 5) response = reject_reason：0 → 1，非 0 → 0
#
# write_txt 预期结构：
# [
#   [user_id, seq_len],
#   problems,        # question_id 序列
#   skills,          # KC 序列
#   answers,         # 答对序列
#   start_time,      # 这里为 NA
#   response_cost    # 这里为 NA
# ]
# =============================================================================

# KC: current_pitches
KEYS = ["user_id", "sequence_id"]


def sanitize_field(x, sep="^"):
    """
    将字段转成字符串，并把逗号替换为 sep，避免和 write_txt 的逗号分隔冲突。
    同时移除换行、制表符和空格。
    """
    s = str(x)
    s = s.replace(",", sep)
    s = s.replace("\n", "").replace("\r", "").replace("\t", "").replace(" ", "")
    return s


def question_id_to_kc(question_id: str) -> str:
    """
    从 question_id 中抽取 current_pitches 部分作为 KC。

    question_id 格式：
        prev_pitches|prev_strings|current_pitches|current_strings

    其中 current_pitches 是第 3 段，即 parts[2]。
    """
    parts = question_id.split("|")
    if len(parts) < 3:
        return question_id
    return parts[2]


def _events_to_event_rows(events_data_str):
    """
    从 events_data 字符串解析出事件列表。

    返回：
        List[(event_idx, question_id, kc, response)]

    question_id 在每个 exercise(record) 内计算：
        prev_pitches|prev_strings|current_pitches|current_strings

    每条 exercise 的起始音：
        prev_pitches = "inf"
        prev_strings = "inf"
    """
    data = json.loads(events_data_str)
    inner = data.get("data", data)

    duration = inner["duration"]
    pitches = inner["pitches"]
    strings = inner["strings"]
    reject_reason = inner["reject_reason"]

    n = len(pitches)
    assert n == len(duration) == len(strings) == len(reject_reason)

    out = []

    for i in range(n):
        current_pitch_s = sanitize_field(pitches[i], sep="^")
        current_string_s = sanitize_field(strings[i], sep="^")

        prev_pitch_s = "inf" if i == 0 else sanitize_field(pitches[i - 1], sep="^")
        prev_string_s = "inf" if i == 0 else sanitize_field(strings[i - 1], sep="^")

        qid = f"{prev_pitch_s}|{prev_string_s}|{current_pitch_s}|{current_string_s}"

        kc = question_id_to_kc(qid)

        # reject_reason: 0 表示正确，非 0 表示错误
        resp = 1 if str(reject_reason[i]) == "0" else 0

        out.append((i, qid, kc, resp))

    return out


def read_data_from_json(read_file, write_file):
    stares = []

    with open(read_file, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    # 1) 只保留 play_mode == "play"
    records = [r for r in raw_list if r.get("play_mode") == "play"]

    if not records:
        raise ValueError("No records with play_mode=='play'")

    rows = []

    for record_idx, r in enumerate(records):
        uid = r["user_id"]
        days = r["days_since_signup"]
        part = r.get("exercise_part_index", 0)
        sess = r.get("session_index", 0)

        try:
            event_list = _events_to_event_rows(r["events_data"])
        except (KeyError, TypeError, json.JSONDecodeError, AssertionError):
            continue

        for event_idx, qid, kc, resp in event_list:
            rows.append(
                {
                    "user_id": uid,
                    "days_since_signup": days,
                    "exercise_part_index": part,
                    "session_index": sess,
                    "record_idx": record_idx,
                    "event_idx": event_idx,
                    "question_id": qid,
                    "sequence_id": kc,
                    "correct": resp,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No valid event rows after parsing events_data")

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)

    print(
        f"after filter play_mode=play, "
        f"interaction num: {ins}, "
        f"user num: {us}, "
        f"question num: {qs}, "
        f"concept num: {cs}, "
        f"avg(ins) per s: {avgins}, "
        f"avg(c) per q: {avgcq}, "
        f"na: {na}"
    )

    user_inters = []

    for user, grp in df.groupby("user_id", sort=False):
        tmp = grp.sort_values(
            by=[
                "days_since_signup",
                "session_index",
                "exercise_part_index",
                "record_idx",
                "event_idx",
            ],
            kind="mergesort",
        )

        seq_problems = tmp["question_id"].astype(str).tolist()
        seq_skills = tmp["sequence_id"].astype(str).tolist()
        seq_ans = tmp["correct"].astype(str).tolist()

        seq_len = len(seq_problems)

        seq_start_time = ["NA"]
        seq_response_cost = ["NA"]

        user_inters.append(
            [
                [str(user), str(seq_len)],
                seq_problems,
                seq_skills,
                seq_ans,
                seq_start_time,
                seq_response_cost,
            ]
        )

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return