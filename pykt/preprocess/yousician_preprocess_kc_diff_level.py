import json
from pathlib import Path
import pandas as pd
from .utils import sta_infos, write_txt
from .yousician_weightnet_infer import (
    FingeringState,
    default_weightnet_ckpt_path,
    load_weightnet_bundle,
    predict_per_note_difficulties,
)
# KC: difficulty_level

KEYS = ["user_id", "sequence_id"]

# =============================================================================
# yousician JSON 预处理
# -----------------------------------------------------------------------------
# 1) 只保留 play_mode == "play" 的样本
# 2) 按 user_id 分组，同一用户内按 days_since_signup / session / part 排序后拼接
# 3) question_id = prev_fret4 | 升序 strings | curr_fret4；首帧 prev 用 0^0^0^0（序列起点无左手把位）
#    与 WeightNet 中 transition(prev,curr) 所需信息一致；跳过帧时 prev 取上一保留帧的 fret
# 4) KC（sequence_id）= WeightNet 对整条 exercise 序列推理得到的逐帧难度，四舍五入为整数 difficult_level
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
    """兼容旧接口：当前 KC 由 WeightNet 逐帧给出，不再从 question_id 派生。"""
    return question_id

def sanitize_field(x, sep="^"):
    """
    将字段转成字符串，并把逗号替换为 sep，避免和 write_txt 的逗号分隔冲突。
    """
    s = str(x)
    s = s.replace(",", sep)
    s = s.replace("\n", "").replace("\r", "").replace("\t", "").replace(' ', '')
    return s

OPEN_MIDI_BY_STRING = {0: 69, 1: 64, 2: 60, 3: 67}

def _to_int_list(x):
    """将单值/列表/字符串（可含 []、逗号、^）统一解析成 int 列表。"""
    def _parse_int_tokens(tokens):
        out = []
        for v in tokens:
            sv = str(v).strip()
            if not sv:
                continue
            low = sv.lower()
            # 自动跳过脏值，避免预处理因单个异常 token 中断
            if low in {"none", "null", "nan"}:
                continue
            try:
                out.append(int(sv))
            except (TypeError, ValueError):
                # 兼容 "3.0" 这类字符串；其余非法值直接跳过
                try:
                    out.append(int(float(sv)))
                except (TypeError, ValueError):
                    continue
        return out

    if isinstance(x, list):
        return _parse_int_tokens(x)
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.replace(",", "^")
    if not s:
        return []
    return _parse_int_tokens(s.split("^"))

def _cumulative_pitch_list(x):
    """pitches 采用首项为 base、后续逐步累加的编码。"""
    vals = _to_int_list(x)
    if not vals:
        return []
    out = [vals[0]]
    for d in vals[1:]:
        out.append(out[-1] + d)
    return out

def _fret4_from_pitch_and_string(pitch_x, string_x):
    """
    将 pitches + strings 映射为固定 4 维把位（按弦 0,1,2,3）。
    未出现弦补 0，返回形如 [3^0^0^0] 的字符串。
    """
    mids = _cumulative_pitch_list(pitch_x)
    strs = _to_int_list(string_x)
    if len(mids) != len(strs):
        raise ValueError(f"音高数与弦数不一致: {mids!r} vs {strs!r}")
    fret4 = [0, 0, 0, 0]
    for m, s in zip(mids, strs):
        if s not in OPEN_MIDI_BY_STRING:
            raise ValueError(f"非法弦号 {s}（应为 0-3）")
        fret4[s] = int(m) - OPEN_MIDI_BY_STRING[s]
    return "[" + "^".join(str(v) for v in fret4) + "]"


def _fret4_list_from_pitch_and_string(pitch_x, string_x):
    """与 _fret4_from_pitch_and_string 相同逻辑，返回长度为 4 的 int 列表（弦 0..3）。"""
    mids = _cumulative_pitch_list(pitch_x)
    strs = _to_int_list(string_x)
    if len(mids) != len(strs):
        raise ValueError(f"音高数与弦数不一致: {mids!r} vs {strs!r}")
    fret4 = [0, 0, 0, 0]
    for m, s in zip(mids, strs):
        if s not in OPEN_MIDI_BY_STRING:
            raise ValueError(f"非法弦号 {s}（应为 0-3）")
        fret4[s] = int(m) - OPEN_MIDI_BY_STRING[s]
    return fret4


def _format_question_id_prev_strings_curr_fret(
    prev_fret4: list, strings_sorted: list, curr_fret4: list
) -> str:
    """唯一标识一步「转移 + 当前发声」：前一帧四弦把位 | 当前帧升序 strings | 当前帧四弦把位。"""
    p = "^".join(str(int(x)) for x in prev_fret4)
    s_part = "^".join(str(int(x)) for x in strings_sorted)
    c = "^".join(str(int(x)) for x in curr_fret4)
    return f"{p}|{s_part}|{c}"


def _events_to_event_rows(events_data_str, weightnet_bundle):
    """从 events_data 解析事件列表 (question_id, kc, response)。
    同一条 exercise 内所有有效帧组成序列，调用 WeightNet 得到逐帧难度 kc（四舍五入整数）。"""
    data = json.loads(events_data_str)
    inner = data.get("data", data)

    duration = inner["duration"]
    pitches = inner["pitches"]
    strings = inner["strings"]
    reject_reason = inner["reject_reason"]
    n = len(pitches)
    if n != len(duration) or n != len(strings) or n != len(reject_reason):
        raise ValueError(
            f"长度不一致: pitches={n}, duration={len(duration)}, "
            f"strings={len(strings)}, reject_reason={len(reject_reason)}"
        )

    model, device, only_played = weightnet_bundle

    row_resp = []
    states_seq = []
    for i in range(n):
        strs_raw = _to_int_list(strings[i])
        strings_sorted = sorted(set(strs_raw))
        if not strings_sorted:
            continue
        try:
            fret4 = _fret4_list_from_pitch_and_string(pitches[i], strings[i])
        except ValueError:
            continue
        try:
            st = FingeringState.from_lists(fret4, strings_sorted)
        except ValueError:
            continue
        resp = 1 if reject_reason[i] == 0 else 0
        row_resp.append(resp)
        states_seq.append(st)

    if not states_seq:
        return []

    # question_id：与模型里相邻两帧转移一致；首帧前一状态为「空把位」0^0^0^0
    qids = []
    for j in range(len(states_seq)):
        prev_fret = list(states_seq[j - 1].fret) if j > 0 else [0, 0, 0, 0]
        curr_fret = list(states_seq[j].fret)
        strings_sorted = list(states_seq[j].strings)
        qid = sanitize_field(
            _format_question_id_prev_strings_curr_fret(prev_fret, strings_sorted, curr_fret)
        )
        qids.append(qid)

    _, per = predict_per_note_difficulties(
        model, states_seq, device, only_played_strings=only_played
    )
    kc_ints = per.round().long().cpu().tolist()
    out = [(qids[j], str(kc_ints[j]), row_resp[j]) for j in range(len(states_seq))]
    return out



def read_data_from_json(read_file, write_file, weightnet_ckpt=None):
    stares = []

    ckpt_path = (
        Path(weightnet_ckpt)
        if weightnet_ckpt is not None
        else default_weightnet_ckpt_path()
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"未找到 WeightNet 权重文件: {ckpt_path}，请训练并保存 weightnet_song_difficulty.pt 或传入 weightnet_ckpt"
        )
    weightnet_bundle = load_weightnet_bundle(ckpt_path)

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
            event_list = _events_to_event_rows(r["events_data"], weightnet_bundle)
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