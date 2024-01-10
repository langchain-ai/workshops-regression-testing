from typing import Tuple
import langsmith
import langsmith.env
from chain import chain as code_langchain_stuff
from langchain.smith import RunEvalConfig
import uuid

import pandas as pd


eval_config = RunEvalConfig(
    # evaluators=["qa"],
)


def get_project_name() -> str:
    git_info = langsmith.env.get_git_info()
    branch, commit = git_info["branch"], git_info["commit"]
    return f"code-langchain-{branch}-{commit[:4]}-{uuid.uuid4().hex[:4]}"


def run_eval() -> Tuple[pd.DataFrame, str]:
    client = langsmith.Client()
    project_name = get_project_name()
    test_results = client.run_on_dataset(
        dataset_name="code-langchain-eval",
        llm_or_chain_factory=lambda: (lambda x: x["question"]) | code_langchain_stuff,
        project_name=project_name,
        evaluation=eval_config,
        verbose=True,
        project_metadata={"context": "regression-tests"},
    )
    return test_results.get_aggregate_feedback(), project_name

def write_output(test_results: pd.DataFrame, project_name: str) -> None:
    client = langsmith.Client()
    project = client.read_project(project_name=project_name)
    dataset = client.read_dataset(dataset_name="code-langchain-eval")
    comparison_url = dataset.url + f"/compare?selectedSessions={project.id}"
    cols = ["execution_time", "error"] + [col for col in test_results.columns if col.startswith("feedback.")]
    markdown_report = test_results[cols]["mean"].to_markdown(
        tablefmt="github"
    )
    with open("report.md", "w") as file:
        file.write(f"## Review results for {project_name}]({comparison_url})\n")
        file.write(markdown_report)

def main():
    test_results, project_name = run_eval()
    write_output(test_results, project_name)
    # assert test_results["feedback.correctness"]["mean"] >= 0.9


if __name__ == "__main__":
    main()
