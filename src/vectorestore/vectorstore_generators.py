from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import CSVLoader


class vectorstorePipeline:
    def __init__(
            self,
            embeddings=OpenAIEmbeddings,
            vectorstore_generator=FAISS
            ) -> None:
        self.embeddings = embeddings
        self.vectorstore_generator = vectorstore_generator

    def __call__(
            self,
            path: str
            ) -> FAISS:

        loader = CSVLoader(path)
        data = loader.load()
        data = [doc.page_content for doc in data]
        return FAISS.from_texts(data, self.embeddings())
