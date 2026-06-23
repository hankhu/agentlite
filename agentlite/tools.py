"""
AgentLite — 工具系统

内置工具注册表 + 常用工具实现。
工具采用 OpenAI Function Calling 兼容的 Schema。
"""

import glob as glob_mod
import logging
import os
import subprocess
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional

# 抑制 urllib3 LibreSSL 旧版 SSL 警告（"urllib3 v2 only supports OpenSSL 1.1.1+"）
warnings.filterwarnings("ignore", message=".*OpenSSL 1\\.1\\.1\\+.*")

from .config import ToolConfig
from .types import ToolCall, ToolDef, ToolResult

logger = logging.getLogger("agentlite.tools")


# ── Tool Registry ───────────────────────────────────────────

class ToolRegistry:
    """工具注册表"""

    def __init__(self, config: Optional[ToolConfig] = None):
        self.config = config or ToolConfig()
        self._tools: Dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        """注册工具"""
        if self.config.allowed and tool.name not in self.config.allowed:
            logger.warning(f"Tool '{tool.name}' not in allowlist, skipping")
            return
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolDef]:
        return self._tools.get(name)

    def list_all(self) -> List[ToolDef]:
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def to_openai_schema(self) -> List[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """执行工具调用"""
        tool = self.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                output=None,
                error=f"Unknown tool: {tool_call.name}. "
                       f"Available: {self.list_names()}",
            )
        try:
            output = tool.function(**tool_call.arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=True,
                output=output,
            )
        except Exception as e:
            logger.exception(f"Tool '{tool_call.name}' failed")
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                output=None,
                error=str(e),
            )


# ── Built-in Tools ──────────────────────────────────────────

def _tool_read_file(path: str, offset: int = 0,
                    limit: Optional[int] = None) -> str:
    """读取文件内容。"""
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        if limit is not None:
            lines = lines[offset:offset + limit]
        else:
            lines = lines[offset:]
        content = "".join(lines)
        if offset > 0 or (limit is not None and limit < total):
            header = f"[File: {path} | lines {offset}-{offset+len(lines)} "
            header += f"of {total}]\n"
            return header + content
        return f"[File: {path}]\n{content}"
    except Exception as e:
        return f"Error reading {path}: {e}"


def _tool_write_file(path: str, content: str, append: bool = False) -> str:
    """写入文件。默认覆盖。"""
    try:
        mode = "a" if append else "w"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        action = "Appended to" if append else "Wrote"
        return f"{action} {path} ({len(content)} chars)"
    except Exception as e:
        return f"Error writing {path}: {e}"


def _tool_list_dir(path: str = ".") -> str:
    """列出目录内容。"""
    if not os.path.isdir(path):
        return f"Error: not a directory: {path}"
    try:
        entries = os.listdir(path)
        entries.sort()
        lines = [f"[Contents of {path}]"]
        for e in entries:
            full = os.path.join(path, e)
            tag = "/" if os.path.isdir(full) else ""
            try:
                size = os.path.getsize(full) if os.path.isfile(full) else 0
                lines.append(f"  {e}{tag}  ({_fmt_size(size)})")
            except OSError:
                lines.append(f"  {e}{tag}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing {path}: {e}"


def _tool_shell_cmd(command: str, cwd: Optional[str] = None,
                    timeout: int = 60) -> str:
    """执行 Shell 命令。timeout 单位为秒，默认 60。"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout
        if out and len(out) > 10000:
            out = out[:10000] + "\n... [truncated]"
        err = result.stderr
        if err and len(err) > 5000:
            err = err[:5000] + "\n... [truncated]"
        parts = []
        if out:
            parts.append(f"STDOUT:\n{out}")
        if err:
            parts.append(f"STDERR:\n{err}")
        parts.append(f"Exit code: {result.returncode}")
        return "\n".join(parts) if parts else f"Exit code: {result.returncode}"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def _tool_python_exec(code: str) -> str:
    """执行 Python 代码。返回 stdout 或最后一个表达式的值。"""
    import io
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        # 先尝试作为表达式求值
        try:
            result = eval(code, {"__builtins__": __builtins__})
            output = captured.getvalue()
            if output:
                return output.rstrip() + f"\n=> {result!r}"
            return f"=> {result!r}"
        except SyntaxError:
            # 作为语句执行
            exec(code, {"__builtins__": __builtins__})
            output = captured.getvalue()
            return output.rstrip() if output else "(executed, no output)"
    except Exception as e:
        output = captured.getvalue()
        parts = []
        if output:
            parts.append(output.rstrip())
        parts.append(f"Error: {type(e).__name__}: {e}")
        return "\n".join(parts)
    finally:
        sys.stdout = old_stdout


def _tool_search_files(pattern: str, path: str = ".",
                       include: str = "*", max_results: int = 50) -> str:
    """在文件中搜索模式 (grep-like)。pattern 支持正则表达式。"""
    import re
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"

    matches = []
    base = os.path.abspath(path)
    files = glob_mod.glob(os.path.join(base, "**", include), recursive=True)

    for fpath in files:
        if not os.path.isfile(fpath):
            continue
        # 跳过二进制和大文件
        if os.path.getsize(fpath) > 1_000_000:
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        rel = os.path.relpath(fpath, base)
                        matches.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                        if len(matches) >= max_results:
                            break
        except Exception:
            continue
        if len(matches) >= max_results:
            break

    if not matches:
        return f"No matches for pattern '{pattern}' in {path}"
    header = f"[Found {len(matches)} matches for '{pattern}' in {path}]\n"
    return header + "\n".join(matches[:max_results])


# ── Web Search ──────────────────────────────────────────────

def _web_search(query: str, max_results: int = 5) -> str:
    """搜索网络（基于 DuckDuckGo，免费无需 API Key）。
    返回标题+摘要+链接列表。"""
    try:
        from ddgs import DDGS
    except ImportError:
        # fallback: 用 requests + BeautifulSoup 直接爬
        return _web_search_fallback(query, max_results)

    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"[No results for '{query}']"

        lines = [f"[Web search results for: {query}]"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()
            href = r.get("href", "")
            lines.append(f"\n{i}. {title}")
            if body:
                lines.append(f"   {body[:200]}")
            if href:
                lines.append(f"   URL: {href}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"ddgs search failed: {e}, trying fallback")
        return _web_search_fallback(query, max_results)


def _web_search_fallback(query: str, max_results: int = 5) -> str:
    """基于 requests + BeautifulSoup 的 DuckDuckGo 搜索 fallback"""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return ("Web search unavailable. Install: pip install ddgs\n"
                "Or: pip install requests beautifulsoup4")

    headers = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
    }
    params = {"q": query, "ia": "web"}
    try:
        resp = requests.get("https://html.duckduckgo.com/html/",
                            params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        for item in soup.select(".result")[:max_results]:
            title_el = item.select_one(".result__title a")
            snippet_el = item.select_one(".result__snippet")
            if title_el:
                title = title_el.get_text(strip=True)
                href = title_el.get("href", "")
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                results.append((title, snippet, href))

        if not results:
            return f"[No results for '{query}']"

        lines = [f"[Web search results for: {query}]"]
        for i, (title, snippet, href) in enumerate(results, 1):
            lines.append(f"\n{i}. {title}")
            if snippet:
                lines.append(f"   {snippet[:200]}")
            if href:
                lines.append(f"   URL: {href}")
        return "\n".join(lines)
    except Exception as e:
        return f"Web search error: {e}"


def _web_fetch(url: str) -> str:
    """获取网页内容并提取可读文本。"""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return "Web fetch unavailable. Install: pip install requests beautifulsoup4"

    headers = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 移除无用标签
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "noscript", "form"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        content = "\n".join(lines[:200])  # 限制 200 行

        title = soup.title.string.strip() if soup.title else ""
        header = f"[Page: {title}]({url})" if title else f"[Page]({url})"
        return f"{header}\n\n{content[:8000]}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


# ── 构建内置工具集 ──────────────────────────────────────────

def build_builtin_tools(config: ToolConfig) -> List[ToolDef]:
    """返回所有内置工具定义"""
    tools = [
        ToolDef(
            name="read_file",
            description="Read a file from the filesystem. "
                        "Use offset and limit for large files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line offset to start reading from (0-based)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read",
                    },
                },
                "required": ["path"],
            },
            function=_tool_read_file,
            dangerous=False,
        ),
        ToolDef(
            name="write_file",
            description="Write content to a file. Creates parent directories "
                        "if needed. Default mode is overwrite.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, append instead of overwrite",
                    },
                },
                "required": ["path", "content"],
            },
            function=_tool_write_file,
            dangerous=True,
        ),
        ToolDef(
            name="list_dir",
            description="List files and directories in a given path.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Defaults to '.'",
                    },
                },
                "required": [],
            },
            function=_tool_list_dir,
            dangerous=False,
        ),
        ToolDef(
            name="shell_cmd",
            description="Execute a shell command. "
                        "Use with caution. Returns stdout, stderr, and exit code.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory for the command",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 60)",
                    },
                },
                "required": ["command"],
            },
            function=_tool_shell_cmd,
            dangerous=True,
        ),
        ToolDef(
            name="python_exec",
            description="Execute Python code and return the result. "
                        "Use for calculations, data processing, or testing logic.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. "
                                       "Expressions are evaluated; "
                                       "statements are executed.",
                    },
                },
                "required": ["code"],
            },
            function=_tool_python_exec,
            dangerous=True,
        ),
        ToolDef(
            name="search_files",
            description="Search for a regex pattern in files under a path. "
                        "Like grep. Returns file:line matches.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default '.')",
                    },
                    "include": {
                        "type": "string",
                        "description": "Glob pattern for file filtering "
                                       "(default '*')",
                    },
                },
                "required": ["pattern"],
            },
            function=_tool_search_files,
            dangerous=False,
        ),
        ToolDef(
            name="web_search",
            description="Search the web for current information. "
                        "Free (DuckDuckGo), no API key required. "
                        "Use for news, facts, and real-time data.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (supports any language)",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10, default 5)",
                    },
                },
                "required": ["query"],
            },
            function=_web_search,
            dangerous=False,
        ),
        ToolDef(
            name="web_fetch",
            description="Fetch a web page and extract its readable text content. "
                        "Use after web_search to read full articles.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the web page to fetch",
                    },
                },
                "required": ["url"],
            },
            function=_web_fetch,
            dangerous=False,
        ),
    ]
    return tools


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size/1024:.1f}KB"
    else:
        return f"{size/(1024*1024):.1f}MB"
