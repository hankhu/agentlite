#!/usr/bin/env python3
"""AgentLite test / demo"""

import json
import sys
import os

sys.path.insert(0, ".")

from agentlite import Agent


def load_config(path: str = "") -> dict:
    if not path:
        path = os.path.expanduser("~/.config/agentlite/config.json")
    path = os.path.expanduser(path)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("llm", {})
    return {}


agent = Agent(llm_config=load_config())
result = agent.run("列出当前目录下所有 Python 文件")
print(f"\n=== Answer ===\n{result}\n=== End ===")
