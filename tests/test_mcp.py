"""Exercise the actual MCP subprocess transport, not only Python function calls."""
import asyncio
import json
import os
import sys
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dqa.core import ROOT


def test_mcp_stdio_handshake_and_tools():
    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "dqa.mcp_server"],
            env={**os.environ, "DQA_HOME": str(ROOT)})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                assert len(listed.tools) == 6
                response = await session.call_tool("list_datasets", {})
                assert not response.isError
                result = await session.call_tool("search_knowledge", {"query": "amount cents"})
                assert not result.isError
                assert any("amount_runbook.md" in c.text for c in result.content if hasattr(c, "text"))
    asyncio.run(asyncio.wait_for(run(), timeout=40))
