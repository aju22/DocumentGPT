import os
import sys
import random
from typing import Any, Dict, Optional, Tuple

import aiohttp
from pydantic import BaseModel, Extra, Field, root_validator

from langchain.utils import get_from_dict_or_env


class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout


class CustomSerpAPIWrapper(BaseModel):
    """Custom Wrapper around SerpAPI.

    This class is used to make requests to the SerpAPI API for fetching search results.
    It also allows you to view the source of the results.

    """

    search_engine: Any  #: :meta private:
    params: dict = Field(
        default={
            "engine": "google",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
        }
    )
    serpapi_api_key: Optional[str] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        serpapi_api_key = get_from_dict_or_env(
            values, "serpapi_api_key", "SERPAPI_API_KEY"
        )
        values["serpapi_api_key"] = serpapi_api_key
        try:
            from serpapi import GoogleSearch

            values["search_engine"] = GoogleSearch
        except ImportError:
            raise ValueError(
                "Could not import serpapi python package. "
                "Please install it with `pip install google-search-results`."
            )
        return values

    async def arun(self, query: str) -> tuple:
        """Run query through SerpAPI and parse result async."""
        return self._process_response(await self.aresults(query))

    def run(self, query: str) -> dict:
        """Run query through SerpAPI and parse result."""
        res = self.results(query)
        toret, idx = self._process_response(res)
        return {"answer": toret, "source_dict": res, "idx": idx}

    def results(self, query: str) -> dict:
        """Run query through SerpAPI and return the raw result."""
        params = self.get_params(query)
        with HiddenPrints():
            search = self.search_engine(params)
            res = search.get_dict()
        return res

    async def aresults(self, query: str) -> dict:
        """Use aiohttp to run query through SerpAPI and return the results async."""

        def construct_url_and_params() -> Tuple[str, Dict[str, str]]:
            param = self.get_params(query)
            param["source"] = "python"
            if self.serpapi_api_key:
                param["serp_api_key"] = self.serpapi_api_key
            param["output"] = "json"
            link = "https://serpapi.com/search"
            return link, param

        url, params = construct_url_and_params()
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    res = await response.json()
        else:
            async with self.aiosession.get(url, params=params) as response:
                res = await response.json()

        return res

    def get_params(self, query: str) -> Dict[str, str]:
        """Get parameters for SerpAPI."""
        _params = {
            "api_key": self.serpapi_api_key,
            "q": query,
        }
        params = {**self.params, **_params}
        return params

    @staticmethod
    def _process_response(res: dict) -> tuple:
        """Process response from SerpAPI."""
        toret = None
        idx = random.randint(0, 3)

        if "error" in res:
            raise ValueError(f"Got error from SerpAPI: {res['error']}")

        elif "organic_results" in res and idx < len(res['organic_results']) and "snippet" in res["organic_results"][
            idx]:
            toret = res["organic_results"][idx]["snippet"]

        return toret, idx
