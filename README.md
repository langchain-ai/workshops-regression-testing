# Regression Testing with LangSmith

LangSmith makes it easy to evaluate your LLM application to ensure that changes to the structure or components don't
create regressions in performance.

The basic process looks like:

1. Create dataset

    a. Upload CSV to LangSmith (we've created a set of Q&A pairs for you: [eval.csv](./data/eval.csv))
    b. You can make further modifications in-app

2. Create eval script (see [test_chain.py](./test_chain.py)), which runs your app over all the example cases in the dataset.
3. Create GHA to automatically run the eval script for each pull request.
4. Review results in LangSmith.


This repository implements a simple chain designed to help answer coding questions about [LangChain Expression Language](https://python.langchain.com/docs/expression_language/). We have added an eval script that runs on every new pull request to track the results over time.