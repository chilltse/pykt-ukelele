import pandas as pd
from .utils import sta_infos, write_txt

KEYS = ["user_id", "sequence_id"]

# =============================================================================
# 参考：这个脚本做什么？
# -----------------------------------------------------------------------------
# 1) 从 CSV 里读出交互数据（每行一般是一条答题/练习记录）
# 2) 统计原始数据的基本信息（交互数、用户数、题目数、知识点数等）
# 3) 清洗数据：去掉关键字段缺失、过滤 correct 非 0/1 的行、把 correct 变成 int
# 4) 再统计一次清洗后的信息
# 5) 按 user_id 分组，把每个用户的交互按时间排序，整理成序列格式
# 6) 写到指定的 txt 文件里（通常是 KT 模型常用的序列输入格式）
# =============================================================================

def read_data_from_csv(read_file, write_file):
    stares = []

    df = pd.read_csv(read_file)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = range(df.shape[0])

    df = df.dropna(subset=["user_id", "log_id", "sequence_id", "correct"])
    df = df[df['correct'].isin([0,1])]#filter responses
    df['correct'] = df['correct'].astype(int)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    ui_df = df.groupby('user_id', sort=False)

    user_inters = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=["log_id", "index"])
        seq_len = len(tmp_inter)
        seq_skills = tmp_inter['sequence_id'].astype(str)
        seq_ans = tmp_inter['correct'].astype(str)
        seq_problems = ["NA"]
        seq_start_time = ["NA"]
        seq_response_cost = ["NA"]

        assert seq_len == len(seq_skills) == len(seq_ans)

        user_inters.append(
            [[str(user), str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_response_cost])

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return

