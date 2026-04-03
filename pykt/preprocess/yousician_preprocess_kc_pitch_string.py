import json
import pandas as pd
from .utils import sta_infos, write_txt

# KC: pitches | strings

KEYS = ["user_id", "sequence_id"]

# =============================================================================
# yousician JSON 预处理
# -----------------------------------------------------------------------------
# 1) 只保留 play_mode == "play" 的样本
# 2) 按 user_id 分组，同一用户内按 days_since_signup / session / part 排序后拼接
# 3) question_id = prev_duration|pitches|strings；prev_duration 按 exercise 判断：该 exercise 的起始音为 "inf"，否则为上一个音的 duration（不跨 exercise 拼接后再算）
# 4) KC 通过 question_id_to_kc(question_id) 得到，默认实现为抽取 pitches 部分
# 5) response = reject_reason：0→1，非0→0


# 将一个用户的数据按 write_txt 预期结构组织：
# [
#   [user_id, seq_len],   # 元信息（一般写在第一行/第一段）
#   problems,             # question_id 序列
#   skills,               # KC 序列
#   answers,              # 答对序列
#   start_time,           # 开始时间序列（这里 NA）
#   response_cost         # 耗时序列（这里 NA）
# ]
# =============================================================================

def question_id_to_kc(question_id: str) -> str:
    """从 question_id 中抽取 pitches 部分作为 KC。question_id 格式: prev_duration|pitches|strings"""
    parts = question_id.split("|", 2)
    if len(parts) < 2:
        return question_id
    return parts[1]

def sanitize_field(x, sep="^"):
    """
    将字段转成字符串，并把逗号替换为 sep，避免和 write_txt 的逗号分隔冲突。
    """
    s = str(x)
    s = s.replace(",", sep)
    s = s.replace("\n", "").replace("\r", "").replace("\t", "").replace(' ', '')
    return s

def _events_to_event_rows(events_data_str):
    """从 events_data 字符串解析出事件列表：每项 (question_id, kc, response)。
    prev_duration 按“每个 exercise(record)”内计算：起始音为 "inf"，否则为上一音的 duration。"""
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
        # ✅ 每条 record（即一个 exercise/part）内起始音 -> inf
        prev_dur = "inf" if i == 0 else str(int(duration[i - 1]))
        # 替换 pitches，strings 里的逗号
        pitch_s = sanitize_field(str(pitches[i]), sep="^")
        string_s = sanitize_field(str(strings[i]), sep="^")

        qid = f"{prev_dur}|{pitch_s}|{string_s}"

        kc = question_id_to_kc(qid)
        resp = 1 if reject_reason[i] == 0 else 0
        out.append((qid, kc, resp))
    return out



def read_data_from_json(read_file, write_file):
    stares = []

    with open(read_file, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    # 1) 只保留 play_mode == "play"
    records = [r for r in raw_list if r.get("play_mode") == "play"]
    if not records:
        raise ValueError("No records with play_mode=='play'")

    # 展平成每行一个事件：(user_id, days, part, sess, question_id, sequence_id, correct)，question_id/kc 已在每条 exercise 内算好
    rows = []
    for r in records:
        uid = r["user_id"]
        days = r["days_since_signup"]
        part = r.get("exercise_part_index", 0)
        sess = r.get("session_index", 0)
        try:
            event_list = _events_to_event_rows(r["events_data"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        for qid, kc, resp in event_list:
            rows.append({
                "user_id": uid,
                "days_since_signup": days,
                "exercise_part_index": part,
                "session_index": sess,
                "question_id": qid,
                "sequence_id": kc,
                "correct": resp,
            })

    df = pd.DataFrame(rows)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after filter play_mode=play, interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    # 2) 按 user_id 分组；3) 按时间排序后直接拼接（question_id / KC 已在每条 exercise 内算好，不在此处再算）
    user_inters = []
    for user, grp in df.groupby("user_id", sort=False):
        tmp = grp.sort_values(by=["days_since_signup", "session_index", "exercise_part_index"])
        seq_problems = tmp["question_id"].astype(str).tolist()
        seq_skills = tmp["sequence_id"].astype(str).tolist()
        seq_ans = tmp["correct"].astype(str).tolist()
        seq_len = len(seq_problems)
        seq_start_time = ["NA"]
        seq_response_cost = ["NA"]
        user_inters.append(
            [[str(user), str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_response_cost]
        )

    write_txt(write_file, user_inters)
    print("\n".join(stares))
    return