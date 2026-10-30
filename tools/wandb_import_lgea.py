"""Đưa log huấn luyện LGEA đã chạy trên Kaggle (output của notebook) lên W&B.

Không chạy lại mô hình. Mỗi lần `python main.py ...` trong notebook thành một run W&B.
Dùng:  python wandb_import_lgea.py --dry-run     # chỉ in ra các run đọc được
       python wandb_import_lgea.py               # đẩy lên W&B
"""
import argparse, json, re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # graphML/
NOTEBOOKS = [
    ROOT / "LGEA/notebooks/reproduce-lgea-emnlp-2025-dbp15k-zh-en.ipynb",
    ROOT / "LGEA/notebooks/reproduce-lgea-emnlp-2025-dbp15k-fr-en.ipynb",
    ROOT / "LGEA/notebooks/reproduce-lgea-table-2-table-3-fbdb15k-fbyg.ipynb",
]
ENTITY = "chaungochuy0709-university-of-science-vietnam-national-"
PROJECT = "graphml-lgea"

ACC_RE = re.compile(r"Ep (\d+) \| (l2r|r2l): acc of top \[1, 10, 50\] = \[([\d. ]+)\], mr = ([\d.]+), mrr = ([\d.]+), Loss = ([\d.]+)")
ARG_RE = re.compile(r"--(\w+) ('?)([^ ']+)\2")


def cell_text(nb, idx):
    c = nb["cells"][idx]
    out = []
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            out.append("".join(o["text"]))
        elif "data" in o:
            out.append("".join(o["data"].get("text/plain", "")))
    return unicodedata.normalize("NFC", "\n".join(out))


def parse_runs(text):
    """Tách output thành từng run theo dòng 'Running command:'.

    Mỗi dòng ACC là một epoch. Khối 'Test result' có thể xuất hiện nhiều lần trong một run
    (FBDB15K: một lần ở epoch 500 trước khi bật huấn luyện lặp, một lần ở epoch 999); lấy khối cuối.
    """
    lines = [l for raw in text.splitlines() for l in raw.split("\r")]
    runs, cur, pending = [], None, None
    for l in lines:
        if "Running command:" in l:
            if cur:
                runs.append(cur)
            args = dict((k, v) for k, _, v in ARG_RE.findall(l.split("Running command:")[1]))
            cur = {"args": args, "epochs": {}, "test": {}, "test_epoch": None}
            pending = None
            continue
        if cur is None:
            continue
        if "Test result" in l:
            pending = {}
            continue
        m = ACC_RE.search(l)
        if m:
            ep, direction, acc, mr, mrr, loss = m.groups()
            vals = [float(x) for x in acc.split()]
            rec = {"hits1": vals[0], "hits10": vals[1], "hits50": vals[2], "mr": float(mr), "mrr": float(mrr), "loss": float(loss)}
            cur["epochs"].setdefault(int(ep), {})[direction] = rec
            if pending is not None:
                pending[direction] = rec
                cur["test"] = pending
                cur["test_epoch"] = int(ep)
                if len(pending) == 2:
                    pending = None
    if cur:
        runs.append(cur)
    return runs


RUN_NAMES = {"LGEA_repro": "LGEA", "LGEA_LGEA": "LGEA (w/o mask)", "LGEA_wo_img": "w/o img",
             "LGEA_wo_rel": "w/o rel", "LGEA_wo_structure": "w/o structure", "LGEA_wo_att": "w/o att",
             "LGEA_wo_llm": "w/o llm", "LGEA_wo_it": "w/o it"}


def run_name(a):
    ds = a.get("data_choice", "")
    split = a.get("data_split", "")
    exp = a.get("exp_name", "")
    label = RUN_NAMES.get(exp, exp)
    return f"{ds} {split} | {label}"


def describe(nb_path, runs):
    out = []
    for r in runs:
        a = r["args"]
        name = run_name(a)
        out.append({"name": name, "dataset": a.get("data_choice"), "split": a.get("data_split"),
                    "rate": a.get("data_rate"), "epochs_logged": len(r["epochs"]),
                    "final_l2r": r["test"].get("l2r"), "final_r2l": r["test"].get("r2l"), "test_epoch": r["test_epoch"],
                    "source": nb_path.name, "args": a})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    all_runs = []
    for p in NOTEBOOKS:
        nb = json.loads(p.read_text(encoding="utf-8"))
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            text = cell_text(nb, i)
            if "Running command:" not in text:
                continue
            for r in parse_runs(text):
                r["_source"] = p.name
                r["_cell"] = i
                all_runs.append(r)
    summary = describe(ROOT, all_runs)
    for s in summary:
        f = s["final_l2r"]
        print(f"{s['source'][:34]:34} cell {s['name']:38} epochs={s['epochs_logged']:4} "
              f"final l2r H@1={f['hits1'] if f else '-'} H@10={f['hits10'] if f else '-'} MRR={f['mrr'] if f else '-'}")
    print(f"\n{len(summary)} run đọc được")
    if a.dry_run:
        return
    import wandb
    for r, s in zip(all_runs, summary):
        cfg = dict(s["args"]); cfg.update({"imported_from_kaggle_output": True, "source_notebook": s["source"], "paper": "LGEA (EMNLP 2025)"})
        run = wandb.init(entity=ENTITY, project=PROJECT, name=s["name"], config=cfg, tags=["imported", "LGEA"],
                         notes=f"Log nhập lại từ output notebook {s['source']} (cell {r['_cell']}), không chạy lại. Ngày chạy gốc ghi trong config.")
        for ep in sorted(r["epochs"]):
            rec = r["epochs"][ep].get("l2r")
            if rec:
                wandb.log({"epoch": ep, "train/loss": rec["loss"], "val/hits1": rec["hits1"], "val/hits10": rec["hits10"], "val/mrr": rec["mrr"]}, step=ep)
        for d in ("l2r", "r2l"):
            t = r["test"].get(d)
            if t:
                run.summary[f"test_{d}/hits1"] = t["hits1"]
                run.summary[f"test_{d}/hits10"] = t["hits10"]
                run.summary[f"test_{d}/mrr"] = t["mrr"]
        wandb.finish()
        print("đã đẩy:", s["name"])


if __name__ == "__main__":
    main()
