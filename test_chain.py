from typing import Optional
import langsmith
import langsmith.env
from chain import chain as code_langchain_stuff
from langchain.smith import RunEvalConfig
from langsmith.evaluation import run_evaluator
from langsmith.schemas import Run, Example
import langsmith
import langsmith.env
import uuid

import pandas as pd

import re


# Heuristic evaluators
@run_evaluator
def contains_code_if_expected(run: Run, example: Optional[Example] = None) -> bool:
    # Simple evaluator that checks if the output contains a markdown code block
    # if the example also contains a markdown code block
    score = (
        int("```" in run.outputs["output"]) if "```" in example.outputs["output"] else 1
    )
    return {
        "name": "contains_code_if_expected",
        "score": score,
    }


@run_evaluator
def doesnt_use_experimental(run: Run, example: Optional[Example] = None) -> bool:
    # Simple evaluator that checks if the output doesn't import
    # from langchain.experimental
    return {
        "name": "doesnt_use_experimental",
        "score": int(
            re.search("langchain.experimental", run.outputs["output"]) is None
        ),
    }


@run_evaluator
def not_empty(run: Run, example: Optional[Example] = None) -> bool:
    # Simple evaluator that checks if the output is not empty
    return {
        "name": "not_empty",
        "score": int(len(run.outputs["output"]) > 0),
    }


eval_config = RunEvalConfig(
    evaluators=["qa"],
    custom_evaluators=[contains_code_if_expected, doesnt_use_experimental, not_empty],
)


def get_project_name() -> str:
    git_info = langsmith.env.get_git_info()
    branch, commit = git_info["branch"], git_info["commit"]
    return f"code-langchain-{branch}-{commit[:4]}-{uuid.uuid4().hex[:4]}"


def run_eval() -> pd.DataFrame:
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
    return test_results.get_aggregate_feedback()


def main():
    test_results = run_eval()
    # We are willing to accept < 100% "correctness", as judged by an LLM
    assert test_results["feedback.correctness"]["mean"] >= 0.9
    # We expect the following to be 100% correct
    assert test_results["feedback.contains_code_if_expected"]["mean"] == 1.0
    assert test_results["feedback.doesnt_use_experimental"]["mean"] == 1.0
    assert test_results["feedback.not_empty"]["mean"] == 1.0


if __name__ == "__main__":
    main()
