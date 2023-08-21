from bs4 import BeautifulSoup
from typing import Union, Any, List
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


class SingleWebLoader(WebBaseLoader):
    """
    This is a custom WebBaseLoader class for extracting text from a web page
    """
    def _scrape(self, url: str, parser: Union[str, None] = None) -> Any:
        """
        Scrapes the content of a web page and returns it as a
        BeautifulSoup object.

        Args:
            url (str): The URL of the web page to scrape.
            parser (Union[str, None], optional): The parser to use for
            BeautifulSoup. If None, it automatically determines the parser
            based on the URL's extension. Defaults to None.

        Returns:
            Any: A BeautifulSoup object containing the parsed HTML content
            of the web page.
        """

        if parser is None:
            if url.endswith(".xml"):
                parser = "xml"
            else:
                parser = self.default_parser

        self._check_parser(parser)

        html_doc = self.session.get(url, **self.requests_kwargs)
        if self.raise_for_status:
            html_doc.raise_for_status()
        html_doc.encoding = html_doc.apparent_encoding

        soup = BeautifulSoup(html_doc.text, parser)

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Remove list elements (ul and ol)
        elements_to_delete = ["ul", "ol", "li", "nav", "address",
                              "span", "svg", "footer"]
        for element in soup.find_all(elements_to_delete):
            element.extract()

        return soup

    def clean_load(self) -> List[Document]:
        """
        Loads documents from a web page, cleans the content, and splits
        into smaller chunks.

        Returns:
            List[Document]: A list of Document objects, each containing
            cleaned and split content.
        """
        documents = self.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=200,
        )
        documents = text_splitter.split_documents(documents)

        for doc in documents:
            doc.page_content = doc.page_content \
                .replace("\n\n\n", "").replace("\t", "")
        return documents
