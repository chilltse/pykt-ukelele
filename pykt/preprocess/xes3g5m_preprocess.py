import pandas as pd
from .utils import write_txt


KEYS = ["uid", "concepts", "questions"]


def _split_seq(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "":
        return []
    return [t.strip() for t in s.split(",")]


def _valid_response(v):
    try:
        iv = int(v)
        return iv in (0, 1)
    except Exception:
        return False


def _ts_sort_value(ts):
    if ts is None:
        return 10**30
    s = str(ts).strip()
    if s == "" or s.upper() == "NA" or s == "-1":
        return 10**30
    try:
        return int(float(s))
    except Exception:
        return 10**30


def read_data_from_csv(read_file, write_file):
    """
    XES3G5M preprocessing (typically from question_level/train_valid_sequences_quelevel.csv):
    - remove padded positions by selectmasks == -1
    - keep only valid responses in {0, 1}
    - write pyKT standard 6-line block format to data.txt
    """
    df = pd.read_csv(read_file, encoding="utf-8", low_memory=False)

    required_cols = {"uid", "questions", "concepts", "responses"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {read_file}: {sorted(missing)}")

    has_timestamps = "timestamps" in df.columns
    has_selectmasks = "selectmasks" in df.columns

    user_inter = []
    total_interactions = 0
    valid_rows = 0

    uniq_users = set()
    uniq_questions = set()
    uniq_concepts_raw = set()
    uniq_concepts_leaf = set()
    user_chunks = dict()

    for ridx, row in df.iterrows():
        uid = str(row["uid"])
        questions = _split_seq(row["questions"])
        concepts = _split_seq(row["concepts"])
        responses = _split_seq(row["responses"])
        timestamps = _split_seq(row["timestamps"]) if has_timestamps else []
        selectmasks = _split_seq(row["selectmasks"]) if has_selectmasks else []

        n = min(len(questions), len(concepts), len(responses))
        if n == 0:
            continue

        q2, c2, r2, t2 = [], [], [], []
        for i in range(n):
            if has_selectmasks and i < len(selectmasks) and selectmasks[i] == "-1":
                continue
            if not _valid_response(responses[i]):
                continue
            q2.append(questions[i])
            c2.append(concepts[i])
            r2.append(str(int(responses[i])))
            if has_timestamps:
                t2.append(timestamps[i] if i < len(timestamps) else "NA")

        if len(q2) == 0:
            continue

        valid_rows += 1

        uniq_users.add(uid)
        uniq_questions.update(q2)
        uniq_concepts_raw.update(c2)
        for c in c2:
            for leaf in str(c).split("_"):
                leaf = leaf.strip()
                if leaf and leaf not in {"-1", "NA"}:
                    uniq_concepts_leaf.add(leaf)
        row_ts_key = _ts_sort_value(t2[0]) if (has_timestamps and len(t2) > 0) else ridx
        user_chunks.setdefault(uid, []).append((row_ts_key, q2, c2, r2, t2))

    # Merge repeated uid rows by timestamp order.
    for uid, chunks in user_chunks.items():
        chunks = sorted(chunks, key=lambda x: x[0])
        merged_q, merged_c, merged_r, merged_t = [], [], [], []
        for _, q2, c2, r2, t2 in chunks:
            merged_q.extend(q2)
            merged_c.extend(c2)
            merged_r.extend(r2)
            if has_timestamps:
                merged_t.extend(t2)

        seq_len = len(merged_q)
        if seq_len == 0:
            continue
        total_interactions += seq_len
        user_inter.append(
            [
                [uid, str(seq_len)],
                [str(x) for x in merged_q],
                [str(x) for x in merged_c],
                [str(x) for x in merged_r],
                [str(x) for x in merged_t] if has_timestamps else ["NA"],
                ["NA"],
            ]
        )

    write_txt(write_file, user_inter)

    avg_ins = round(total_interactions / len(uniq_users), 4) if uniq_users else 0.0
    print(
        "after xes3g5m preprocess, "
        f"interaction num: {total_interactions}, "
        f"user num: {len(uniq_users)}, "
        f"question num: {len(uniq_questions)}, "
        f"concept num (leaf): {len(uniq_concepts_leaf)}, "
        f"concept num (raw token): {len(uniq_concepts_raw)}, "
        f"avg(ins) per s: {avg_ins}, "
        f"seq rows kept: {valid_rows}, "
        f"uid merged sequences: {len(user_inter)}"
    )
