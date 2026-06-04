import ast
import re

import pandas as pd

from openai_batch_wrapper.batch_manager import BatchManager
from src.reporting.excel import ExcelReporter

JOB_STEM   = "pilot_batch"
INPUT_PATH = "to_batch_pilot"
OUTPUT_PATH = "output"
TIERS = ["tier1", "tier2"]

BASE_COLS          = ["filingId", "companyid", "companyname", "filingDate",
                      "chunks", "chunk_ids", "prompt_version", "model_x"]
SHARED_METRIC_COLS = ["model_y", "prompt_tokens", "completion_tokens", "notes"]

_ILLEGAL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


# ── data cleaning / parsing ───────────────────────────────────────────────────

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(
        lambda col: col.map(lambda v: _ILLEGAL_CHARS.sub("", v) if isinstance(v, str) else v)
        if col.dtype == object else col
    )


def _is_dict_col(series: pd.Series) -> bool:
    sample = series.dropna()
    if sample.empty:
        return False
    v = sample.iloc[0]
    return isinstance(v, dict) or (isinstance(v, str) and v.strip().startswith("{"))


def _expand_dict_col(series: pd.Series, prefix: str) -> pd.DataFrame:
    def parse(v):
        if pd.isna(v) or v == "":
            return {}
        if isinstance(v, dict):
            return v
        try:
            return ast.literal_eval(str(v))
        except Exception:
            return {}

    parsed = series.apply(parse)
    seen: dict[str, None] = {}
    for d in parsed:
        seen.update(dict.fromkeys(d.keys()))

    rows: dict[str, list] = {}
    for key in seen:
        rows[f"{prefix}_{key}_adopted"] = parsed.apply(lambda d, k=key: d.get(k, {}).get("adopted"))
        rows[f"{prefix}_{key}_quote"]   = parsed.apply(lambda d, k=key: d.get(k, {}).get("quote"))
        rows[f"{prefix}_{key}_page"]    = parsed.apply(lambda d, k=key: d.get(k, {}).get("page"))

    return pd.DataFrame(rows, index=series.index)


# ── batch status + fetch ──────────────────────────────────────────────────────

def check_and_fetch(tier: str) -> str:
    job_id = f"{JOB_STEM}_{tier}"
    bm = BatchManager(
        job_id=job_id,
        input_jsonl_path=f"{INPUT_PATH}/{job_id}.jsonl",
        batch_task_reset=False,
        output_path=OUTPUT_PATH,
    )
    status, _ = bm.get_batch_status()
    print(f"[{job_id}] status: {status}")
    if status in ("completed", "expired"):
        bm.get_output_file()
        print(f"[{job_id}] output file fetched")
    else:
        print(f"[{job_id}] not ready yet — skipping output fetch")
    bm.close()
    return status


# ── merge + export ────────────────────────────────────────────────────────────

def merge_outputs():
    ref_df    = pd.read_parquet(f"{INPUT_PATH}/{JOB_STEM}_ref.parquet")
    tier_dfs  = {}

    for tier in TIERS:
        job_id     = f"{JOB_STEM}_{tier}"
        output_csv = f"{OUTPUT_PATH}/output_{job_id}.csv"
        try:
            output_df = pd.read_csv(output_csv, dtype={"custom_id": "string"})
        except FileNotFoundError:
            print(f"[{job_id}] output CSV not found, skipping")
            continue

        merged = ref_df.merge(
            output_df.rename(columns={"custom_id": "chunk_ids"}),
            on="chunk_ids", how="left",
        )
        merged.to_csv(f"{OUTPUT_PATH}/clean_output_{job_id}.csv", index=False)
        print(f"[{job_id}] wrote {len(merged)} rows ({ref_df['filingId'].nunique()} filings)")
        tier_dfs[tier] = merged

    if tier_dfs:
        _combine_and_export(tier_dfs)


def _combine_and_export(tier_dfs: dict):
    combined = None

    for tier, df in tier_dfs.items():
        tier_specific = [c for c in df.columns if c not in BASE_COLS]
        rename_map    = {c: f"{c}_{tier}" for c in SHARED_METRIC_COLS if c in tier_specific}
        working       = df[BASE_COLS + tier_specific].rename(columns=rename_map)

        final_cols: list[str] = []
        for col in [rename_map.get(c, c) for c in tier_specific]:
            if _is_dict_col(working[col]):
                expanded = _expand_dict_col(working[col], prefix=col)
                working  = pd.concat([working.drop(columns=[col]), expanded], axis=1)
                final_cols.extend(expanded.columns.tolist())
            else:
                final_cols.append(col)

        drop = [c for c in final_cols if c.endswith("_page") or c.startswith("model_")]
        if drop:
            working    = working.drop(columns=drop)
            final_cols = [c for c in final_cols if c not in drop]

        combined = working if combined is None else combined.merge(
            working.drop(columns=[c for c in BASE_COLS if c != "chunk_ids"]),
            on="chunk_ids", how="outer",
        )

    out_path = f"{OUTPUT_PATH}/combined_{JOB_STEM}.xlsx"
    ExcelReporter(_clean_df(combined)).write(out_path)
    print(f"wrote combined Excel ({len(combined)} rows) → {out_path}")


# ── entrypoint ────────────────────────────────────────────────────────────────

for tier in TIERS:
    check_and_fetch(tier)

merge_outputs()
