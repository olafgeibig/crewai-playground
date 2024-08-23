from typing import Optional, Any, Type, Literal
from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool

class SpiderToolSchema(BaseModel):
    url: str = Field(description="Website URL")
    mode: Literal["scrape", "crawl"] = Field(
        default="scrape",
        description="Mode, the only two allowed modes are `scrape` or `crawl`. Use `scrape` to scrape a single page and `crawl` to crawl the entire website following subpages."
    )
    limit: Optional[int] = Field(
        default=None,
        description="The maximum number of pages allowed to crawl per website. Set to 0 or omit to crawl all pages."
    )
    depth: Optional[int] = Field(
        default=None,
        description="The crawl limit for maximum depth. If 0, no limit will be applied."
    )
    metadata: Optional[bool] = Field(
        default=False,
        description="Boolean to include metadata or not."
    )
    query_selector: Optional[str] = Field(
        default=None,
        description="The CSS query selector to use when extracting content from the markup."
    )

class SpiderTool(BaseTool):
    name: str = "Spider scrape & crawl tool"
    description: str = "Scrape & Crawl any url and return LLM-ready data."
    args_schema: Type[BaseModel] = SpiderToolSchema
    api_key: Optional[str] = None
    spider: Optional[Any] = None
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from spider import Spider # type: ignore
        except ImportError:
           raise ImportError(
               "`spider-client` package not found, please run `pip install spider-client`"
           )

        self.spider = Spider(api_key=api_key)

    def _run(
        self,
        url: str,
        mode: Literal["scrape", "crawl"] = "scrape",
        limit: Optional[int] = None,
        depth: Optional[int] = None,
        metadata: bool = False,
        query_selector: Optional[str] = None
    ) -> str:
        params = {
            "return_format": "markdown",
            "limit": limit,
            "depth": depth,
            "metadata": metadata,
            "query_selector": query_selector
        }
        # Remove None values from params
        params = {k: v for k, v in params.items() if v is not None}

        if mode == "scrape":
            spider_docs = self.spider.scrape_url(url=url, **params)
        else:
            spider_docs = self.spider.crawl_url(url=url, **params)

        return spider_docs
