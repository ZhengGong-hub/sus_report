"""Download the finished shards and parse them in one step, instead of running download then parse.

One command, one config read: honors openai_batch.allow_partial (preview mode) end to end. Download
merges the completed shards into aggregated_output_batch_results[.__preview].csv, then parse joins that
onto the batch-input reference and writes parsed_aggregated_batch_output[.__preview].csv.
"""

from carbontax.config import load_run_config
from carbontax.openai_batch import CONFIG_PATH
from carbontax.openai_batch.batch_job import OpenAIBatchJob
from carbontax.openai_batch.parse_output import parse_output
from carbontax.utils.logger import setup_logging


def main() -> None:
    setup_logging()
    cfg = load_run_config(CONFIG_PATH)
    run_name = cfg["run_name"]
    allow_partial = cfg["openai_batch"]["allow_partial"]  # true = preview: completed shards only, *__preview files
    OpenAIBatchJob(run_name).download(allow_partial=allow_partial)  # writes the merged output CSV
    parse_output(run_name, preview=allow_partial)  # reads that CSV, writes the joined parsed CSV


if __name__ == "__main__":
    main()
