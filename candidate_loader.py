"""Stage 2 (part 1) — streaming candidate loader. Avoids holding raw JSON text
twice in memory; yields parsed dicts one at a time so the rest of the pipeline
can build flat feature rows without ever materializing the full raw corpus.
"""
import json
import gzip


def iter_candidates(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
