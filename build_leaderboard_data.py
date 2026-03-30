#!/usr/bin/env python3

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "results_real"
OUTPUT_ROOT = Path(__file__).resolve().parent / "data"
MIRROR_ROOT = OUTPUT_ROOT / "results_real"
LEADERBOARD_JSON = OUTPUT_ROOT / "leaderboard-data.json"


TASK_CONFIG = {
    "K-disentQA": {
        "label": "SCA-QA",
        "metricLabel": "Speech Context Faithfulness",
        "shortMetric": "Faithfulness",
        "lowerBetter": False,
    },
    "SQA": {
        "label": "Speech QA",
        "metricLabel": "Accuracy (%)",
        "shortMetric": "Acc(%)",
        "lowerBetter": False,
    },
    "Instruct": {
        "label": "Speech Instruction",
        "metricLabel": "GPT-4o Judge Score",
        "shortMetric": "GPT-4o Judge",
        "lowerBetter": False,
    },
    "ASR": {
        "label": "ASR",
        "metricLabel": "CER (%)",
        "shortMetric": "CER",
        "lowerBetter": True,
    },
    "Translation": {
        "label": "Translation",
        "metricLabel": "BLEU / METEOR",
        "shortMetric": "BLEU / METEOR",
        "lowerBetter": False,
    },
    "LSQA": {
        "label": "Long Speech Understanding",
        "metricLabel": "Accuracy (%)",
        "shortMetric": "Acc(%)",
        "lowerBetter": False,
    },
}


DATASET_LABELS = {
    "history_before_chosun": "History_before_chosun",
    "history_before_chosun_other": "History_before_chosun Other",
    "history_after_chosun": "History_after_chosun",
    "history_after_chosun_other": "History_after_chosun Other",
    "history_after_chosun_tts": "History_after_chosun TTS",
    "k-sports": "K-sports",
    "k-sports_other": "K-sports Other",
    "kpop": "K-pop",
    "kpop_other": "K-pop Other",
    "click": "CLICk",
    "click_other": "CLICk Other",
    "kobest_boolq": "KoBest BoolQ",
    "kobest_boolq_other": "KoBest BoolQ Other",
    "kudge": "KUDGE",
    "kudge_other": "KUDGE Other",
    "vicuna": "Vicuna",
    "vicuna_other": "Vicuna Other",
    "openhermes": "OpenHermes",
    "openhermes_other": "OpenHermes Other",
    "alpaca": "Alpaca",
    "alpaca_other": "Alpaca Other",
    "ksponspeech_eval_clean": "KsponSpeech Clean",
    "ksponspeech_eval_other": "KsponSpeech Other",
    "common_voice_korea": "CommonVoice-KO",
    "common_voice_korea_noisy": "CommonVoice-KO Noisy",
    "zeroth_korean_test": "Zeroth-Korean",
    "zeroth_korean_test_noisy": "Zeroth-Korean Noisy",
    "etri_tst-COMMON_processed": "ETRI-TST-Common",
    "etri_tst-HE_processed": "ETRI-TST-HE",
    "mctest_final_with_audio-add-evidence-7": "MCTest",
    "mctest_final_with_audio-add-evidence-7_noise": "MCTest Noise",
}


def dataset_label(dataset_id: str) -> str:
    return DATASET_LABELS.get(dataset_id, dataset_id)


def extract_metric(task_name: str, payload: dict):
    if task_name == "K-disentQA":
        value = payload.get("accuracy_speech")
        if value is None:
            return None
        value = value * 100
        return {"value": value, "display": f"{value:.2f}"}

    if task_name == "SQA":
        value = payload.get("accuracy_logit")
        if value is None:
            value = payload.get("accuracy_generation")
        if value is None:
            return None
        value = value * 100
        return {"value": value, "display": f"{value:.2f}"}

    if task_name == "Instruct":
        value = payload.get("avg_gpt_score")
        if value is None:
            return None
        value = value * 100
        return {"value": value, "display": f"{value:.2f}"}

    if task_name == "ASR":
        value = payload.get("total_cer")
        if value is None:
            return None
        value = value * 100
        return {"value": value, "display": f"{value:.2f}"}

    if task_name == "Translation":
        bleu = payload.get("avg_bleu")
        if bleu is None:
            bleu = payload.get("corpus_bleu")
        meteor = payload.get("avg_meteor")
        if bleu is None:
            return None
        if meteor is None:
            return {"value": bleu, "display": f"{bleu:.2f}"}
        return {"value": bleu, "display": f"{bleu:.2f} / {meteor:.2f}"}

    if task_name == "LSQA":
        value = payload.get("accuracy_logit")
        if value is None:
            value = payload.get("accuracy_generation")
        if value is None:
            return None
        value = value * 100
        return {"value": value, "display": f"{value:.2f}"}

    return None


def prefer_model_name(existing: str, candidate: str) -> str:
    if existing:
        return existing
    return candidate or ""


def copy_summary_tree():
    if MIRROR_ROOT.exists():
        shutil.rmtree(MIRROR_ROOT)
    MIRROR_ROOT.mkdir(parents=True, exist_ok=True)

    for summary_path in SOURCE_ROOT.rglob("*_summary.json"):
        relative_path = summary_path.relative_to(SOURCE_ROOT)
        destination = MIRROR_ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_path, destination)


def build_entries():
    entries_by_model = {}
    task_datasets = {task_name: set() for task_name in TASK_CONFIG}

    for task_dir in sorted(SOURCE_ROOT.iterdir()):
        if not task_dir.is_dir() or task_dir.name not in TASK_CONFIG:
            continue

        task_name = task_dir.name
        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            model_id = model_dir.name
            entry = entries_by_model.setdefault(
                model_id,
                {
                    "id": model_id,
                    "rank_name": model_id,
                    "model": "",
                    "url": "",
                    "tasks": {},
                },
            )

            entry["tasks"].setdefault(task_name, {})

            for dataset_dir in sorted(model_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue

                summary_files = [
                    path
                    for path in dataset_dir.glob("*_summary.json")
                    if path.parent == dataset_dir
                ]
                if not summary_files:
                    continue

                summary_path = sorted(summary_files)[0]
                with summary_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)

                metric = extract_metric(task_name, payload)
                if metric is None:
                    continue

                entry["model"] = prefer_model_name(entry["model"], payload.get("model"))
                if not entry["model"]:
                    entry["model"] = model_id

                dataset_id = dataset_dir.name
                entry["tasks"][task_name][dataset_id] = metric
                task_datasets[task_name].add(dataset_id)

    tasks = []
    for task_name, metadata in TASK_CONFIG.items():
        discovered = sorted(task_datasets[task_name])
        tasks.append(
            {
                "id": task_name,
                "label": metadata["label"],
                "metricLabel": metadata["metricLabel"],
                "shortMetric": metadata["shortMetric"],
                "lowerBetter": metadata["lowerBetter"],
                "datasets": [
                    {"id": dataset_id, "label": dataset_label(dataset_id)}
                    for dataset_id in discovered
                ],
            }
        )

    return tasks, list(entries_by_model.values())


def write_leaderboard_data(tasks, entries):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceRoot": "data/results_real",
        "tasks": tasks,
        "entries": entries,
    }
    with LEADERBOARD_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main():
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"Missing source root: {SOURCE_ROOT}")

    copy_summary_tree()
    tasks, entries = build_entries()
    write_leaderboard_data(tasks, entries)
    print(f"Wrote mirrored summaries -> {MIRROR_ROOT}")
    print(f"Wrote leaderboard data -> {LEADERBOARD_JSON}")


if __name__ == "__main__":
    main()
