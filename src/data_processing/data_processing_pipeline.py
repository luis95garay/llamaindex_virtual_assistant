from generators.generators import OpenAIGenerator
from generators.qa_generators import QApipeline
from vectorestore.vectorstore_generators import vectorstorePipeline
from loaders import MAPPED_LOADERS_METHODS
import pickle
from pathlib import Path


class PipelineDataProcessing:
    def __init__(
            self,
            source_type,
            generator=OpenAIGenerator,
            qa_generator=QApipeline,
            vectorstore_generator=vectorstorePipeline()
            ) -> None:
        self.loader = MAPPED_LOADERS_METHODS[source_type]
        self.qa_generator = qa_generator(generator=generator)
        self.vectorstore_generator = vectorstore_generator
        data_path = Path(__file__).parent.parent.parent / "data"
        self.questions_path = data_path / "questions"
        self.vectorstores_path = data_path / "vectorstores"

    def __call__(
            self,
            path: str,
            name: str,
            proceadure_type: str
            ) -> None:
        loader = self.loader(path, verify_ssl=False)
        documents = loader.clean_load()

        for doc in documents:
            df_questions = self.qa_generator(doc.page_content)
            print("generó las preguntas y respuestas")
            # Save csv
            q_save_path = self.questions_path / f"{name}.csv"
            df_questions.to_csv(str(q_save_path), index=False)

            # Create current vectorstore
            current_vectorstore = self.vectorstore_generator(str(q_save_path))

            if proceadure_type == "add":
                try:
                    last_vs_path = self.vectorstores_path / "vectorstore.pkl"
                    with open(str(last_vs_path), "rb") as f:
                        previous_vectorstore = pickle.load(f)
                    current_vectorstore.merge_from(previous_vectorstore)

                except FileNotFoundError:
                    pass
            elif proceadure_type == "update":
                pass

            # Save vectorstore
            with open(str(last_vs_path), "wb") as f:
                pickle.dump(current_vectorstore, f)
