import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

import aiohttp
import chz

from tinker_cookbook.renderers import ToolCall, Message
from tinker_cookbook.renderers.base import ToolSpec

logger = logging.getLogger(__name__)


WEB_SEARCH_TOOL: ToolSpec = {
    "name": "search",
    "description": "Search the web for relevant information with the queries. Returns a list of urls with a snippet of the content in the url for each query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Search queries given to the search engine.",
            }
        },
        "required": ["query_list"],
    }
}

VECTOR_SEARCH_TOOL: ToolSpec = {
    "name": "search",
    "description": "Search the web for relevant information with the queries. Returns a list of urls with a snippet of the content in the url for each query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of fully-formed semantic queries. This tool will return search results for each query.",
            }
        },
        "required": ["query_list"],
    }
}

SINGLE_SEARCH_TOOL: ToolSpec = {
    "name": "search",
    "description": "Search the web for relevant information with the query. Returns a list of urls with a snippet of the content in the url.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query given to the search engine.",
            }
        },
        "required": ["query"],
    }
}



WEB_VISIT_TOOL: ToolSpec = {
    "name": "visit",
    "description": "Browse the web pages by their urls. This returns the text content of the web page, with at most 10000 characters. Optionally, you can search for a specific query in each url, and the tool will perform fuzzy matching to find the part of the page that contains the highest textual similarity to the query.",
    "parameters": {
        "type": "object",
        "properties": {
            "url_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of urls to browse. This tool will return the text content of each page.",
            },
            "query_list": {
                "type": "array",
                "items": {"type": "string"},
                "nullable": True,
                "description": "A list of queries to search for in each url. The tool will perform fuzzy matching to find the part of the page that contains the highest textual similarity to the query. If given an empty query, the tool will return the beginning of the page.",
            }
        },
        "required": ["url_list"],
        "additionalProperties": False
    },
    "outputSchema": {
        "type": "string",
        "description": "The browse results in JSON format",
    },
}

VECTOR_VISIT_TOOL: ToolSpec = {
    "name": "visit",
    "description": "Browse the web pages by their urls. The available urls are limited to the ones returned by the search tool. Make sure to use the exact url from the search tool or it will not be available. This returns the text content of the web page, with at most 10000 characters. Optionally, you can search for a specific query in each url, and the tool will perform fuzzy matching to find the part of the page that contains the highest textual similarity to the query.",
    "parameters": {
        "type": "object",
        "properties": {
            "url_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of urls to browse. This tool will return the text content of each page.",
            },
            "query_list": {
                "type": "array",
                "nullable": True,
                "items": {"type": "string"},
                "description": "A list of queries to search for in each url. The tool will perform fuzzy matching to find the part of the page that contains the highest textual similarity to the query. If given an empty query, the tool will return the beginning of the page.",
            }
        },
        "required": ["url_list"],
        "additionalProperties": False
    },
    "outputSchema": {
        "type": "string",
        "description": "The browse results in JSON format",
    },
}

SINGLE_VISIT_TOOL: ToolSpec = {
    "name": "visit",
    "description": "Browse the web page by its url. This returns the text content of the web page, with at most 10000 characters. Optionally, you can search for a specific query in the url, and the tool will perform fuzzy matching to find the part of the page that contains the highest textual similarity to the query.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Url of the web page to browse. This tool will return the text content of the page.",
            },
            "query": {
                "type": "string",
                "nullable": True,
                "description": "Query to search for in the url. The tool will perform fuzzy matching to find the part of the page that contains the highest textual similarity to the query. If given an empty query, the tool will return the beginning of the page.",
            }
        },
        "required": ["url"],
        "additionalProperties": False
    },
    "outputSchema": {
        "type": "string",
        "description": "The browse result in JSON format",
    },
}


class ToolClientInterface(ABC):
    @abstractmethod
    def get_tool_schemas(self) -> list[ToolSpec]: ...

    @abstractmethod
    async def invoke(self, tool_call: ToolCall) -> list[Message]: ...


@chz.chz
class WebSearchToolConfig:
    port: int = 8000
    topk: int = 10
    content_length: int = 10000
    scoring_func: str = "rouge"
    chunking_func: str = "newline"
    timeout: float = 300.0  # Timeout in seconds (default 5 minutes)
    search_mode: Literal["default", "vector", "single"] = "default"


class WebSearchTool(ToolClientInterface):
    def __init__(self, config: WebSearchToolConfig):
        self.url = f"http://localhost:{config.port}"
        self.topk = config.topk
        self.content_length = config.content_length
        self.scoring_func = config.scoring_func
        self.chunking_func = config.chunking_func
        self.search_mode = config.search_mode
        self.timeout = aiohttp.ClientTimeout(total=config.timeout)

    def get_tool_schemas(self) -> list[ToolSpec]:
        if self.search_mode == "vector":
            return [VECTOR_SEARCH_TOOL, VECTOR_VISIT_TOOL]
        elif self.search_mode == "single":
            return [SINGLE_SEARCH_TOOL, SINGLE_VISIT_TOOL]
        else:
            return [WEB_SEARCH_TOOL, WEB_VISIT_TOOL]

    def validate_and_extract_search_args(self, args: dict) -> tuple[list[str] | None, str | None]:
        """Validate and extract query list from search tool arguments.

        Returns (query_list, error_message). If error_message is not None, the args are invalid.
        """
        if self.search_mode == "single":
            if "query" not in args:
                return None, f"Invalid argument in tool calls: {args}\nMake sure to include the required 'query' argument when using the search tool."
            return [args["query"]], None
        else:
            if "query_list" not in args:
                return None, f"Invalid argument in tool calls: {args}\nMake sure to include the required 'query_list' argument when using the search tool."
            return args["query_list"], None

    def validate_and_extract_visit_args(self, args: dict) -> tuple[tuple[list[str], list[str] | None] | None, str | None]:
        """Validate and extract url list and query list from visit tool arguments.

        Returns ((url_list, query_list), error_message). If error_message is not None, the args are invalid.
        """
        if self.search_mode == "single":
            if "url" not in args:
                return None, f"Invalid argument in tool calls: {args}\nMake sure to include the required 'url' argument when using the visit tool."
            return ([args["url"]], [args.get("query", "")]), None
        else:
            if "url_list" not in args:
                return None, f"Invalid argument in tool calls: {args}\nMake sure to include the required 'url_list' argument when using the visit tool."
            return (args["url_list"], args.get("query_list")), None


    async def batch_search(self, query_list: list[str]) -> str:
        if len(query_list) == 0 or not isinstance(query_list, list) or not all(isinstance(query, str) and query.strip() for query in query_list):
            return json.dumps({"error": "Please provide a list of queries to search for."})

        # use aiohttp and asyncio.gather to call the /search endpoint for each query concurrently
        async def search_wrapper(session: aiohttp.ClientSession, query: str) -> dict:
            payload = {"query": query, "topk": self.topk}
            async with session.post(self.url + "/search", params=payload) as response:
                if response.status != 200:
                    return {
                        "query": query,
                        "output": f"Error: {response.status}"
                    }
                result = await response.json()
                output = result['output']['formatted_output']
                return {
                    "query": query,
                    "output": output
                }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # better to use the batch search endpoint instead of calling the search endpoint for each query
            payload = {"query": query_list, "topk": self.topk}
            async with session.post(self.url + "/search_batch", params=payload) as response:
                if response.status != 200:
                    return {
                        "query_list": query_list,
                        "output": f"Error: {response.status}"
                    }
                result = await response.json()
                outputs = result['output']
                outputs = [{"query": query, "output": output["formatted_output"]} for query, output in zip(query_list, outputs)]
                return json.dumps(outputs, indent=2)
            
            # tasks = [search_wrapper(session, query) for query in query_list]
            # results = await asyncio.gather(*tasks)
        
        return json.dumps(results, indent=2)


    async def batch_browse(self, id_list: list[str], query_list: list[str] | None = None) -> str:
        if len(id_list) == 0 or not isinstance(id_list, list) or not all(isinstance(id, str) and id.strip() for id in id_list):
            return json.dumps({"error": "Please provide a non-empty list of ids to browse."})

        if query_list is None or len(query_list) == 0:
            query_list = [""] * len(id_list)

        # the query list can be None (treat as no query for all urls) and each query can be empty (treat as no query for the url)
        if not isinstance(query_list, list) or not all(isinstance(query, str) for query in query_list):
            return json.dumps({"error": "Please provide a list of queries to search for."})

        async def browse_wrapper(session: aiohttp.ClientSession, id: str, query: str) -> dict:
            payload = {
                "id": id, "query": query, 
                "content_length": self.content_length, 
                "scoring_func": self.scoring_func, 
                "chunking_func": self.chunking_func
            }
            async with session.post(self.url + "/open_url", params=payload) as response:
                # check if success, if not then return error message
                if response.status != 200:
                    return {
                        "id": id,
                        "query": query,
                        "output": f"Error: {response.status}"
                    }
                result = await response.json()
                output = result['output']
                return {
                    "id": id,
                    "query": query,
                    "output": output
                }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            tasks = [browse_wrapper(session, id, query) for id, query in zip(id_list, query_list)]
            results = await asyncio.gather(*tasks)
        
        return json.dumps(results, indent=2)


    async def invoke(self, tool_call: ToolCall) -> list[Message]:
        if tool_call.function.name == "search":
            query_list, error = self.validate_and_extract_search_args(tool_call.function.arguments)
            if error:
                return [Message(role="tool", content=error)]
            output = await self.batch_search(query_list)
            return [Message(role="tool", content=output)]
        elif tool_call.function.name == "visit":
            visit_args, error = self.validate_and_extract_visit_args(tool_call.function.arguments)
            if error:
                return [Message(role="tool", content=error)]
            id_list, query_list = visit_args
            output = await self.batch_browse(id_list, query_list)
            return [Message(role="tool", content=output)]
        else:
            return [Message(role="tool", content=f"Invalid tool name: {tool_call.function.name}")]
