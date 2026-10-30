"""Đưa kết quả MAKAR (FMNERG, GMNER) đã chạy trên Kaggle lên W&B. Không chạy lại mô hình.

Dùng:  python wandb_import_makar.py --dry-run                 # chỉ in ra
       python wandb_import_makar.py --dataset GMNER          # đẩy một bộ lên project graphml-makar
       python wandb_import_makar.py                          # đẩy cả FMNERG và GMNER
"""
import argparse, json, re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "MAKAR/notebooks/makar-reproduce-kaggle.ipynb"
GMNER_LOG = ROOT / "MAKAR/logs/makar-gmner-reproduce.log"  # log kernel Kaggle, DATASET="GMNER"
ENTITY = "chaungochuy0709-university-of-science-vietnam-national-"
PROJECT = "graphml-makar"
EP_RE = re.compile(r'^\d\d:\d\d (kea|nokea) (\{"epoch".*\})$', re.M)


def out(nb, i):
    c = nb["cells"][i]
    t = ""
    for o in c.get("outputs", []):
        t += "".join(o.get("text", "")) if o.get("output_type") == "stream" else "".join(o.get("data", {}).get("text/plain", ""))
    return unicodedata.normalize("NFC", t)


def epoch_runs(nb):
    """Log NER theo epoch: hai run (KEA, w/o KEA). Mỗi epoch xuất hiện nhiều lần trong log, giữ một lần."""
    t = out(nb, 13)
    runs = {"KEA": {}, "w/o KEA": {}}
    for k, j in EP_RE.findall(t):
        d = json.loads(j)
        name = "KEA" if k == "kea" else "w/o KEA"
        runs[name].setdefault(d["epoch"], d)
    return runs


def gmner_epoch_runs():
    """Log NER theo epoch của lần chạy GMNER, đọc từ log kernel (dòng JSON stream)."""
    t = unicodedata.normalize("NFC", "".join(e.get("data", "") for e in json.loads(GMNER_LOG.read_text(encoding="utf-8"))))
    runs = {"KEA": {}, "w/o KEA": {}}
    for k, j in EP_RE.findall(t):
        d = json.loads(j)
        runs["KEA" if k == "kea" else "w/o KEA"].setdefault(d["epoch"], d)
    return runs


# Kết quả cuối (FMNERG, Bảng 3–5 của báo cáo). Nguồn: output cell 22, 24, 30 của notebook.
SUMMARY = [
    {"name": "MAKAR (KEA+ECA+ERGA)", "MNERG": 63.75, "MNER": 77.22, "EEG": 73.57, "paper_MNERG": 60.54, "paper_MNER": 71.24, "paper_EEG": 75.66},
    {"name": "w/o ECA", "MNERG": 63.51, "MNER": 76.93, "EEG": 73.57, "paper_MNERG": 58.57},
    {"name": "w/o KEA", "MNERG": 56.00, "MNER": 67.59, "EEG": 71.94, "paper_MNERG": 57.50},
    {"name": "Raw Qwen2.5-VL-3B (300 câu)", "EEG": 23.54, "paper_EEG": 38.38},
    {"name": "MAKAR-3B GRPO (300 câu)", "MNERG": 64.09, "MNER": 78.14, "EEG": 74.98, "paper_EEG": 72.43},
]

# Kết quả cuối GMNER (kernel huychau/makar-gmner-reproduce, 2026-10-30). Nguồn: results_GMNER.json trong output kernel.
SUMMARY_GMNER = [
    {"name": "MAKAR (KEA+ECA+ERGA)", "MNERG": 69.29, "MNER": 84.37, "EEG": 72.58, "paper_MNERG": 71.88, "paper_MNER": 86.38, "paper_EEG": 74.64, "paper_EEG_3B": 72.48},
    {"name": "w/o ECA", "MNERG": 69.24, "MNER": 84.32, "EEG": 72.53, "paper_MNERG": 69.23},
    {"name": "w/o KEA", "MNERG": 67.87, "MNER": 82.54, "EEG": 71.63, "paper_MNERG": 67.63},
    {"name": "Raw Qwen2.5-VL-3B (300 câu)", "EEG": 22.95, "paper_EEG": 22.45},
    {"name": "MAKAR-3B GRPO (300 câu)", "MNERG": 69.24, "MNER": 85.66, "EEG": 72.60, "paper_EEG": 72.48},
]


def push(dataset, er, summary, source, date):
    import wandb
    for k, v in er.items():
        run = wandb.init(entity=ENTITY, project=PROJECT, name=f"{dataset} | NER {k}",
                         config={"dataset": f"Twitter-{dataset}", "stage": "NER", "kea": k == "KEA", "epochs": 8,
                                 "source": source, "imported_from_kaggle_output": True,
                                 "paper": "MAKAR (EMNLP 2025)"},
                         tags=["imported", "MAKAR"],
                         notes=f"Nhập lại từ output Kaggle ({date}), không chạy lại.")
        for ep in sorted(v):
            d = v[ep]
            wandb.log({"epoch": ep, "train/loss": d["loss"], "val/dev_f1": d["dev_f1"]}, step=ep)
        best = max(v.values(), key=lambda d: d["dev_f1"])
        run.summary["best_dev_f1"] = best["dev_f1"]
        print("đã đẩy NER", dataset, k, run.id)
        wandb.finish()
    for s in summary:
        cfg = {"dataset": f"Twitter-{dataset}", "system": s["name"], "source": source,
               "imported_from_kaggle_output": True, "paper": "MAKAR (EMNLP 2025)"}
        run = wandb.init(entity=ENTITY, project=PROJECT, name=f"{dataset} | {s['name']}", config=cfg,
                         tags=["imported", "MAKAR", "results"], notes=f"Số cuối nhập từ output Kaggle ({date}), không chạy lại.")
        for m in ("MNERG", "MNER", "EEG", "paper_MNERG", "paper_MNER", "paper_EEG", "paper_EEG_3B"):
            if m in s:
                run.summary[m] = s[m]
        print("đã đẩy", dataset, s["name"], run.id)
        wandb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dataset", choices=["FMNERG", "GMNER", "all"], default="all")
    a = ap.parse_args()
    jobs = []
    if a.dataset in ("FMNERG", "all"):
        jobs.append(("FMNERG", epoch_runs(json.loads(NB.read_text(encoding="utf-8"))), SUMMARY, NB.name, "2026-10-13"))
    if a.dataset in ("GMNER", "all"):
        jobs.append(("GMNER", gmner_epoch_runs(), SUMMARY_GMNER, GMNER_LOG.name, "2026-10-30"))
    for ds, er, summary, src, _ in jobs:
        for k, v in er.items():
            print(f"{ds} NER run {k}: {len(v)} epoch, dev F1 cuối = {max(v)} -> {v[max(v)]['dev_f1']:.2f}")
        print(f"{ds} kết quả cuối: {len(summary)} run")
    if a.dry_run:
        return
    for job in jobs:
        push(*job)


if __name__ == "__main__":
    main()
