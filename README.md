# Data Quality Investigation Assistant

A local Python MVP for investigating CSV data quality with scikit-learn, retrieval, and an MCP server.


<img width="1320" height="574" alt="image" src="https://github.com/user-attachments/assets/d6e22fc5-7f8c-42ad-a87c-5a4162782663" />


## Quick start on Windows

Open a terminal in this folder and run:

    setup.cmd
    start.cmd

Open http://127.0.0.1:8765. Select orders_baseline.csv and orders_current.csv,
then click **Investigate changes**. API documentation is at http://127.0.0.1:8765/docs.
Stop a foreground server with Ctrl+C.

If setup has already been completed, only start.cmd is needed.
The project uses a private .venv environment; Python 3.11+ and internet for initial dependency installation are required.

## What is implemented

- CSV inventory and profiling: inferred schema, nulls, unique values, exact duplicate rows, finite numeric statistics.
- Baseline comparison: row count, added/removed columns, inferred type changes, missingness and numeric mean shifts.
- IsolationForest trained only on baseline rows, then applied to the current dataset. Default feature selection excludes id and *_id.
- Local TF-IDF passage retrieval over Markdown/text schemas, lineage notes, incident notes and runbooks.
- Evidence-only investigation works immediately without an API key or model download.
- Optional local Ollama RAG uses retrieved passages and calculated measurements to generate an explanation.
- Six read-only MCP tools let a compatible AI host retrieve and synthesize the same evidence.
- Browser dashboard and downloadable investigation JSON.

## Your own data

Copy UTF-8, comma-separated CSV files with headers into data/. Use one healthy baseline and one current export of the same table.
Copy Markdown or UTF-8 text documents into knowledge/. Refresh the page to see new CSV files.
Knowledge is reindexed on each search, so edits are visible immediately.
DQA_HOME can override the root containing data/ and knowledge/.
Never commit real customer exports; CSV files are ignored by the supplied .gitignore.

The demo generator is reproducible and does not overwrite existing files:

    .venv\Scripts\python.exe -m dqa.demo

The synthetic current dataset contains 20 amount values multiplied by 100, 40 missing regions,
15 extended delivery values, and 10 appended duplicate rows. These known changes are demonstration fixtures,
not evidence that a particular production incident happened.

## Enable local RAG generation

Install Ollama separately and download a model suitable for your hardware using its official instructions.
In PowerShell, set the name of a model you already installed, then start the app:

    $env:DQA_OLLAMA_MODEL = "your-installed-model-name"
    .\start.cmd

Enable **Use my local Ollama model** in the dashboard.
The app calls only http://127.0.0.1:11434/api/chat for generation.
If no model is configured or the request fails, it returns computed evidence and an explicit fallback message.
The model is not bundled or downloaded by setup. Generated explanations can be wrong; inspect their cited evidence.

Without Ollama, the dashboard performs profiling and retrieval, not LLM generation.
An MCP host can perform retrieval-augmented generation itself using returned evidence.

## MCP connection

mcp-config.example.json contains the exact local Python path and DQA_HOME for this installation.
Copy its server entry into an MCP-compatible client's configuration.
Configuration formats vary by client; this example uses the common mcpServers JSON format.
The server uses stdio. It is launched by the client, independently from the web dashboard.

Tools:
- list_datasets()
- profile_dataset(dataset)
- compare_datasets(current, baseline)
- detect_anomalies(current, baseline, columns=None)
- search_knowledge(query, limit=5)
- investigate_dataset(current, baseline, question)

Example request to your AI host:

> Compare orders_current.csv with orders_baseline.csv. Investigate amount spikes and missing regions.
> Cite the runbooks, distinguish observations from hypotheses, and suggest checks without changing data.

MCP returns data to the client you connect. A cloud-backed host may send that evidence to its model provider.
The server itself has no database write, arbitrary SQL or remediation tools.

## Architecture

Browser -> FastAPI -> shared profiling / comparison / IsolationForest services
                       -> TF-IDF retrieval -> optional local Ollama model
MCP client -> stdio MCP server -> the same shared services

Key files:
- dqa/core.py: dataset bounds, profiling, comparison, anomaly detection.
- dqa/retrieval.py: chunking, retrieval, evidence assembly, optional model generation.
- dqa/api.py: REST API and dashboard route.
- dqa/mcp_server.py: MCP tools.
- dqa/static/index.html: self-contained dashboard.
- knowledge/: sample schema, lineage and investigation runbooks.
- tests/: analytical, API, RAG fallback and real MCP subprocess tests.

## Run tests

    .venv\Scripts\python.exe -m pytest -q

The model integration test uses a stub: it checks prompt assembly and response handling without requiring Ollama.
The MCP test performs a real subprocess initialize/list_tools/call_tool sequence.

## MVP boundaries and next development steps

This is a single-user local prototype, not a production warehouse service.
CSV files are loaded into memory and limited to 50 MB and 100,000 rows each.
Document files above 1 MB are skipped. Knowledge corpora should stay small; retrieval is rebuilt per request.
No persistent vector database, database connector, scheduled monitoring, authentication or tenant isolation is implemented.
The server binds to loopback only. Do not expose it publicly.

TF-IDF matches terms; it is not semantic embedding retrieval.
The lineage in this demo is supplied documentation, not automatically discovered.
Outlier scores are not probabilities or confirmed errors. Baselines must be representative.
Missing numeric values are imputed using baseline medians; inspect missingness separately.
The automatic contamination threshold may flag valid rows; domain rules and threshold evaluation are future work.
Mean shifts do not detect every distribution change, account for seasonality, or prove a root cause.
CSV types are inferred. Numeric IDs other than id / *_id should be excluded explicitly through the MCP columns argument.

Recommended next milestones:
1. Read-only PostgreSQL connector and per-table dataset registry.
2. Persisted profiling history, scheduled runs and data contracts.
3. Hybrid lexical/embedding retrieval with versioned document ingestion.
4. Authentication, per-source access controls and audit logging.
5. Evaluation on labeled incidents, temporal baselines and alert threshold tuning.

## Implementation references

- Official MCP Python SDK v1: https://py.sdk.modelcontextprotocol.io/v1/
  The dependency is pinned below version 2 to match the FastMCP API used here.
- IsolationForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- Ollama chat API: https://docs.ollama.com/api/chat
