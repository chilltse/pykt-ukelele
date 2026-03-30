import os
import argparse
import json
import copy
import torch
import pandas as pd

from pykt.models import evaluate,evaluate_question,load_model
from pykt.datasets import init_test_datasets

device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

with open("../configs/wandb.json") as fin:
    wandb_config = json.load(fin)

def _load_uid_list(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        if "uid" not in df.columns:
            return None
        if "fold" in df.columns:
            df = df[df["fold"] == -1]
        return df["uid"].tolist()
    except Exception as e:
        print(f"[warn] failed to load uid list from {file_path}: {e}")
        return None

def dump_deep_irt_factors(model, data_loader, model_name, save_dir, split_name="test", uid_list=None, data_config=None):
    """
    只保存 DeepIRT 的 stu_ability 和 que_diff（有效时间步，按 smasks 过滤后展平）。
    输出:
      - {emb_type}_{split_name}_stu_ability.pt
      - {emb_type}_{split_name}_que_diff.pt
    """
    if model_name != "deep_irt":
        return

    model.eval()
    all_ability = []
    all_diff = []
    rows = []
    sample_offset = 0

    with torch.no_grad():
        for data in data_loader:
            dcur = data
            q, c, r = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"]
            qshft, cshft, rshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"]
            m, sm = dcur["masks"], dcur["smasks"]

            q, c, r = q.to(device), c.to(device), r.to(device)
            qshft, cshft, rshft = qshft.to(device), cshft.to(device), rshft.to(device)
            m, sm = m.to(device), sm.to(device)

            # 与 evaluate() 保持一致：deep_irt 输入拼接后的 cc / cr
            cc = torch.cat((c[:, 0:1], cshft), dim=1)
            cr = torch.cat((r[:, 0:1], rshft), dim=1)

            # qtest=True -> 返回 p, f, k
            _, f, k = model(cc.long(), cr.long(), qtest=True)

            # 对齐评估位置（去掉第0列）
            f = f[:, 1:, :]  # [B, T, D]
            k = k[:, 1:, :]  # [B, T, D]

            # 计算两项
            stu_ability = model.ability_layer(model.dropout_layer(f)).squeeze(-1)  # [B, T]
            que_diff = model.diff_layer(model.dropout_layer(k)).squeeze(-1)         # [B, T]

            # 仅保留有效时间步（和 evaluate 一致）
            ability_valid = torch.masked_select(stu_ability, sm).detach().cpu()
            diff_valid = torch.masked_select(que_diff, sm).detach().cpu()

            all_ability.append(ability_valid)
            all_diff.append(diff_valid)

            cshft_cpu = cshft.detach().cpu()
            sm_cpu = sm.detach().cpu()
            ability_cpu = stu_ability.detach().cpu()
            diff_cpu = que_diff.detach().cpu()
            batch_size = ability_cpu.shape[0]
            seq_len = ability_cpu.shape[1]

            for i in range(batch_size):
                seq_idx = sample_offset + i
                uid = uid_list[seq_idx] if (uid_list is not None and seq_idx < len(uid_list)) else seq_idx
                for t in range(seq_len):
                    if sm_cpu[i, t]:
                        rows.append({
                            "uid": uid,
                            "t": int(t),
                            "kc_id": int(cshft_cpu[i, t].item()),
                            "stu_ability": float(ability_cpu[i, t].item()),
                            "que_diff": float(diff_cpu[i, t].item()),
                        })
            sample_offset += batch_size

    all_ability = torch.cat(all_ability, dim=0) if len(all_ability) > 0 else torch.empty(0)
    all_diff = torch.cat(all_diff, dim=0) if len(all_diff) > 0 else torch.empty(0)

    stu_path = os.path.join(save_dir, f"{model.emb_type}_{split_name}_stu_ability.pt")
    diff_path = os.path.join(save_dir, f"{model.emb_type}_{split_name}_que_diff.pt")

    torch.save(all_ability, stu_path)
    torch.save(all_diff, diff_path)

    print(f"[dump] saved stu_ability -> {stu_path}, shape={tuple(all_ability.shape)}")
    print(f"[dump] saved que_diff    -> {diff_path}, shape={tuple(all_diff.shape)}")

    if len(rows) == 0:
        return

    trace_df = pd.DataFrame(rows)
    trace_path = os.path.join(save_dir, f"{model.emb_type}_{split_name}_deep_irt_trace.parquet")
    trace_df.to_parquet(trace_path, index=False)
    print(f"[dump] saved trace       -> {trace_path}, rows={len(trace_df)}")

    # 只在 test split 生成你需要的两张汇总表
    if split_name != "test":
        return

    # 1) 所有学生在最后时刻的能力（按 uid）
    last_ability = (
        trace_df.sort_values(["uid", "t"])
        .groupby("uid", as_index=False)
        .tail(1)[["uid", "t", "stu_ability"]]
        .sort_values("uid")
        .reset_index(drop=True)
    )
    last_ability_path = os.path.join(save_dir, f"{model.emb_type}_students_last_ability.csv")
    last_ability.to_csv(last_ability_path, index=False, encoding="utf-8-sig")
    print(f"[dump] saved student last ability -> {last_ability_path}")

    # 2) 所有 KC 难度排行（按 que_diff 均值降序）
    kc_rank = (
        trace_df.groupby("kc_id", as_index=False)["que_diff"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "que_diff_mean", "median": "que_diff_median", "count": "n_obs"})
        .sort_values("que_diff_mean", ascending=False)
        .reset_index(drop=True)
    )

    concept_id2name = {}
    if data_config is not None:
        keyid2idx_path = os.path.join(data_config["dpath"], "keyid2idx.json")
        if os.path.exists(keyid2idx_path):
            try:
                with open(keyid2idx_path, "r", encoding="utf-8") as f:
                    keyid2idx = json.load(f)
                concept_id2name = {int(v): k for k, v in keyid2idx["concepts"].items()}
            except Exception as e:
                print(f"[warn] failed to parse {keyid2idx_path}: {e}")

    kc_rank["kc_name"] = kc_rank["kc_id"].map(concept_id2name).fillna("UNKNOWN")
    kc_rank = kc_rank[["kc_id", "kc_name", "que_diff_mean", "que_diff_median", "n_obs"]]
    kc_rank_path = os.path.join(save_dir, f"{model.emb_type}_kc_difficulty_ranking.csv")
    kc_rank.to_csv(kc_rank_path, index=False, encoding="utf-8-sig")
    print(f"[dump] saved kc difficulty ranking -> {kc_rank_path}")

def main(params):
    if params['use_wandb'] ==1:
        import wandb
        os.environ['WANDB_API_KEY'] = wandb_config["api_key"]
        wandb.init(project="wandb_predict")

    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
            if remove_item in model_config:
                del model_config[remove_item]    
        trained_params = config["params"]
        fold = trained_params["fold"]
        model_name, dataset_name, emb_type = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"]
        if model_name in ["saint", "sakt", "atdkt"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len   

    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]    
    if model_name not in ["dimkt"]:        
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size)
    else:
        diff_level = trained_params["difficult_levels"]
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size, diff_level=diff_level)

    print(f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")

    model = load_model(model_name, model_config, data_config, emb_type, save_dir)
    
    # 先持久化 DeepIRT 的 stu_ability / que_diff（仅 deep_irt 生效）
    test_uid_list = _load_uid_list(os.path.join(data_config["dpath"], data_config["test_file"]))
    test_window_uid_list = _load_uid_list(os.path.join(data_config["dpath"], data_config["test_window_file"]))
    dump_deep_irt_factors(
        model, test_loader, model_name, save_dir, split_name="test",
        uid_list=test_uid_list, data_config=data_config
    )
    dump_deep_irt_factors(
        model, test_window_loader, model_name, save_dir, split_name="test_window",
        uid_list=test_window_uid_list, data_config=data_config
    )


    save_test_path = os.path.join(save_dir, model.emb_type+"_test_predictions.txt")

    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        rel = None
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl" 
            rel = pd.read_pickle(os.path.join(dpath, fname))                

    if model.model_name == "rkt":
        testauc, testacc = evaluate(model, test_loader, model_name, rel, save_test_path)
    else:
        testauc, testacc = evaluate(model, test_loader, model_name, save_test_path)
    print(f"testauc: {testauc}, testacc: {testacc}")

    window_testauc, window_testacc = -1, -1
    save_test_window_path = os.path.join(save_dir, model.emb_type+"_test_window_predictions.txt")
    if model.model_name == "rkt":
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, rel, save_test_window_path)
    else:
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, save_test_window_path)
    print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    # question_testauc, question_testacc = -1, -1
    # question_window_testauc, question_window_testacc = -1, -1
  
    dres = {
        "testauc": testauc, "testacc": testacc, "window_testauc": window_testauc, "window_testacc": window_testacc,
    }  

    q_testaucs, q_testaccs = -1,-1
    qw_testaucs, qw_testaccs = -1,-1
    if "test_question_file" in data_config and not test_question_loader is None:
        save_test_question_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions.txt")
        q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_test_question_path)
        for key in q_testaucs:
            dres["oriauc"+key] = q_testaucs[key]
        for key in q_testaccs:
            dres["oriacc"+key] = q_testaccs[key]
            
    if "test_question_window_file" in data_config and not test_question_window_loader is None:
        save_test_question_window_path = os.path.join(save_dir, model.emb_type+"_test_question_window_predictions.txt")
        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_test_question_window_path)
        for key in qw_testaucs:
            dres["windowauc"+key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres["windowacc"+key] = qw_testaccs[key]

        
    # print(f"testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")
    # print(f"question_testauc: {question_testauc}, question_testacc: {question_testacc}, question_window_testauc: {question_window_testauc}, question_window_testacc: {question_window_testacc}")
    
    print(dres)
    raw_config = json.load(open(os.path.join(save_dir,"config.json")))
    dres.update(raw_config['params'])

    if params['use_wandb'] ==1:
        wandb.log(dres)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="early_fusion,late_fusion")
    parser.add_argument("--use_wandb", type=int, default=1)

    args = parser.parse_args()
    print(args)
    params = vars(args)
    main(params)
