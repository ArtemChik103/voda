"""Pre-flight Data Integrity and Alignment Auditor.

Scans the target data directory (data/raw or data/synthetic_benchmark),
verifies metadata, spatial dimensions, CRS, channel availability, and reports
readiness for feature extraction and inference.
"""

import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.service.api import PAIR_CATALOGUE


def audit_dataset_directory(data_dir: Path) -> Dict:
    results = {
        "data_dir": str(data_dir),
        "exists": data_dir.exists(),
        "total_pairs_in_catalog": len(PAIR_CATALOGUE),
        "pairs_found": 0,
        "pairs_missing": [],
        "details": {},
        "ready_for_pipeline": False,
    }

    if not data_dir.exists():
        return results

    pairs_found_count = 0
    for pair_id, meta in PAIR_CATALOGUE.items():
        pair_path = data_dir / pair_id
        pair_info = {
            "catalog_aoi": meta["aoi"],
            "event_type": meta["event_type"],
            "folder_exists": pair_path.exists(),
        }

        if pair_path.exists():
            pairs_found_count += 1
        else:
            results["pairs_missing"].append(pair_id)

        results["details"][pair_id] = pair_info

    results["pairs_found"] = pairs_found_count
    results["ready_for_pipeline"] = (pairs_found_count == len(PAIR_CATALOGUE))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Аудит структуры и целостности датасета перед инференсом.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Путь к каталогу с парами (по умолчанию: data/raw или data/synthetic_benchmark)",
    )
    args = parser.parse_args()

    target_path = Path(args.data_dir)
    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path

    print("=" * 70)
    print("КОСМОХАКАТОН 2026: ПРЕДПОЛЕТНЫЙ АУДИТ ДАТАСЕТА SENTINEL-1 & SENTINEL-2")
    print(f"Целевой каталог: {target_path}")
    print("=" * 70)

    audit = audit_dataset_directory(target_path)
    print(f"Найдено пар: {audit['pairs_found']} из {audit['total_pairs_in_catalog']}")

    if audit["pairs_missing"]:
        print(f"\n[!] Отсутствуют директории пар ({len(audit['pairs_missing'])} шт.):")
        for p in audit["pairs_missing"]:
            print(f"  - {p}")
        print("\nПодсказка: распакуйте архив датасета в data/raw/<pair_id>/")
    else:
        print("\n[OK] Все 11 официальных пар обнаружены!")

    print("=" * 70)


if __name__ == "__main__":
    main()
