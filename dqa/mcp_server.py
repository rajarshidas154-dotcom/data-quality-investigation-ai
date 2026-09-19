"""Run with python -m dqa.mcp_server; stdout is reserved for MCP."""
from mcp.server.fastmcp import FastMCP
from dqa import core, retrieval

mcp = FastMCP("Data Quality Investigator")


@mcp.tool()
def list_datasets() -> list[str]:
    """List local CSV datasets available for read-only investigation."""
    return core.datasets()


@mcp.tool()
def profile_dataset(dataset: str) -> dict:
    """Measure schema, nulls, duplicate rows and numeric statistics for a local CSV."""
    return core.profile(dataset)


@mcp.tool()
def compare_datasets(current: str, baseline: str) -> dict:
    """Compare schema, row counts, missingness and numeric means against a baseline."""
    return core.compare(current, baseline)


@mcp.tool()
def detect_anomalies(current: str, baseline: str, columns: list[str] | None = None) -> dict:
    """Fit IsolationForest on baseline and score current rows; flags are not confirmed errors."""
    return core.anomalies(current, baseline, columns)


@mcp.tool()
def search_knowledge(query: str, limit: int = 5) -> list[dict]:
    """Retrieve cited passages from local schemas, lineage notes and runbooks."""
    return retrieval.search(query, limit)


@mcp.tool()
def investigate_dataset(current: str, baseline: str, question: str) -> dict:
    """Return computed evidence and retrieved context for the calling AI to synthesize."""
    return retrieval.investigate(current, baseline, question)


if __name__ == "__main__":
    mcp.run(transport="stdio")

