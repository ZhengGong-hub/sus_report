"""
Stratified sample of ~100 companies across simpleindustryid buckets,
restricted to companies that have ESG filings in the mapping.

Writes the sampled company IDs to src/research_config/pilot.yaml,
ready to be passed to run_research().

Usage
-----
  python src/runner/sample_companies.py
"""

import random
import yaml
import pandas as pd

COMPANY_CSV        = "company.csv"
MAPPING_CSV        = "mapping_data/company_esgfiling_mapping.csv"
OUTPUT_CONFIG      = "src/research_config/pilot.yaml"
TARGET_N           = 30
MAX_PER_INDUSTRY   = 3    # cap per simpleindustryid to ensure diversity
RANDOM_SEED        = 42


def sample_companies(
    target_n: int = TARGET_N,
    max_per_industry: int = MAX_PER_INDUSTRY,
    seed: int = RANDOM_SEED,
) -> list[int]:
    company_df = pd.read_csv(COMPANY_CSV, low_memory=False, usecols=["companyid", "simpleindustryid"])
    company_df = company_df.drop_duplicates("companyid").dropna(subset=["simpleindustryid"])

    mapping_df = pd.read_csv(MAPPING_CSV, low_memory=False, usecols=["companyid"])
    mapping_df = mapping_df.drop_duplicates("companyid")

    # restrict to companies that actually have ESG filings
    pool = mapping_df.merge(company_df, on="companyid", how="inner")

    # stratified sample: up to max_per_industry per bucket
    sampled = pd.concat([
        g.sample(min(len(g), max_per_industry), random_state=seed)
        for _, g in pool.groupby("simpleindustryid")
    ])

    # if over target, sample down while preserving industry spread
    if len(sampled) > target_n:
        sampled = sampled.sample(target_n, random_state=seed)

    print(f"Sampled {len(sampled)} companies across {sampled['simpleindustryid'].nunique()} industries")
    print(sampled["simpleindustryid"].value_counts().to_string())

    return sorted(sampled["companyid"].tolist())


def write_config(company_ids: list[int], path: str = OUTPUT_CONFIG) -> None:
    config = {
        "identifier": "companyid",
        "companyid": company_ids,
        "output_folder": "to_batch_pilot",
        "jsonl_path": "pilot_batch",
        "ref_data_path": "pilot_batch_ref",
    }
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\nWrote config to {path}")


if __name__ == "__main__":
    ids = sample_companies()
    write_config(ids)
