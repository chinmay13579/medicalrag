"""Print exactly what would be sent to the LLM for a question (pseudonymised + minimum context).

    python scripts/show_prompt.py "Why did Jana Novakova's lab results not appear?"
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragkit.guardrails.input import check_input  # noqa: E402
from ragkit.pipeline import RagPipeline  # noqa: E402


def main() -> None:
    logging.getLogger("ragkit").setLevel(logging.ERROR)
    q = " ".join(sys.argv[1:]) or "Why did Jana Novakova's lab results not appear?"
    pipe = RagPipeline()

    print(f"RAW QUESTION        : {q}")
    verdict = check_input(q)
    print(f"INPUT GUARD         : {'allow' if verdict.allowed else 'BLOCK ' + str(verdict.reasons)}")
    if not verdict.allowed:
        return
    safe_q, vault, bundle = pipe.build(q)
    print(f"PSEUDONYMISED QUERY : {safe_q}")
    print(f"VAULT (never sent)  : {json.dumps(vault.mapping)}")
    print(f"\n{'=' * 24} SYSTEM {'=' * 24}\n{bundle.system}")
    print(f"\n{'=' * 24} USER {'=' * 26}\n{bundle.user}")
    print(f"\n{'=' * 60}")
    print(f"context tokens ~{bundle.context_tokens} | chunks sent: {len(bundle.items)} | "
          f"dropped: {json.dumps(bundle.dropped)}")
    leaked = pipe.pseudo.residual(bundle.system + bundle.user)
    print(f"residual PII scan   : {'CLEAN' if not leaked else leaked}")


if __name__ == "__main__":
    main()
