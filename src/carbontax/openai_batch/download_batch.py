"""Download the finished batch output. Reads the run named in openai_batch.CONFIG_PATH."""

from carbontax.config import load_run_config
from carbontax.openai_batch import CONFIG_PATH
from carbontax.openai_batch.batch_job import OpenAIBatchJob
from carbontax.utils.logger import setup_logging


def main() -> None:
    setup_logging()
    OpenAIBatchJob(load_run_config(CONFIG_PATH)["run_name"]).download()


if __name__ == "__main__":
    main()
