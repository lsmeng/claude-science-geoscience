#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2.0,<2"]
# ///
"""Start each connector the way Claude Science would (`uv run server.py` over
stdio), list its tools, and make one real call. Needs network.

    uv run tests/test_connectors.py
"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent / "connectors"

# One real call per connector: the 2019 Ridgecrest M7.1 mainshock.
CALLS = {
    "comcat": ("comcat_search", dict(
        starttime="2019-07-06T03:00:00", endtime="2019-07-06T04:00:00", minmagnitude=6.5)),
    "isc": ("isc_search", dict(
        starttime="2019-07-06T03:00:00", endtime="2019-07-06T04:00:00", minmagnitude=6.5)),
    "fdsnws": ("get_events", dict(
        starttime="2019-07-06T03:00:00", endtime="2019-07-06T04:00:00", minmagnitude=6.5,
        datacenter="USGS")),
}


async def run(name):
    params = StdioServerParameters(command="uv", args=["run", str(ROOT / name / "server.py")])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            tool, args = CALLS[name]
            res = await s.call_tool(tool, args)
            text = res.content[0].text
            try:
                d = json.loads(text)
                ev = d.get("events") or []
                summary = f"{d.get('count')} event(s)" + (
                    f", first: {ev[0].get('time')} M{ev[0].get('mag')}" if ev else "")
            except Exception:
                summary = text[:200]
            return tools, tool, summary, bool(res.isError)


async def main():
    bad = 0
    for name in CALLS:
        try:
            tools, tool, summary, err = await run(name)
            status = "FAIL" if err else "PASS"
            bad += err
            print(f"[{status}] {name}: tools={tools}\n        {tool} -> {summary}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    sys.exit(1 if bad else 0)


asyncio.run(main())
