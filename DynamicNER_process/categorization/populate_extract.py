#!/usr/bin/env python3
"""
populate_extract.py

Copia base/<lang>/SWIFT/extract/<split>.jsonl in dynamic/extract/<lang>/<split>.jsonl,
uniformando la struttura a quella di dynamic/classify/<lang>/ (vista nello
screenshot del progetto). A differenza di classify, extract NON passa dalla
pipeline dynamic1-4.py: quel formato (frase -> frase con entita' marcate
##...##) non ha "opzioni" da unire/sostituire/ridurre, quindi non c'e'
nulla su cui applicare merge/sinonimi/riduzione/gestione speciale.

Uso:
    python populate_extract.py --lang it --base-dir ../../DynamicNER
    (di default processa tutti gli split train/dev/test presenti)
"""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="it")
    parser.add_argument("--base-dir", default="DynamicNER")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    src_dir = base_dir / "base" / args.lang / "SWIFT" / "extract"
    dst_dir = base_dir / "dynamic" / "extract" / args.lang
    dst_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        # L'estensione reale puo' essere .jsonl o .json: proviamo entrambe.
        src_candidates = [src_dir / f"{split}.jsonl", src_dir / f"{split}.json"]
        src = next((c for c in src_candidates if c.exists()), None)
        if src is None:
            print(f"[SKIP] {split}: nessun file trovato in {src_dir} "
                  f"(provati .jsonl e .json)")
            continue

        dst = dst_dir / f"{split}.jsonl"
        shutil.copy2(src, dst)

        n_lines = sum(1 for _ in open(src, encoding="utf-8") if _.strip())
        print(f"[OK] {split}: {src} -> {dst} ({n_lines} righe)")


if __name__ == "__main__":
    main()