from .generators import OpenAIGenerator
import json
import pandas as pd


class QApipeline:
    def __init__(
            self,
            generator=OpenAIGenerator
            ) -> None:

        self.generator = generator()

    def generate_questions(self) -> str:
        # Create prompt for question generation
        output = "una única llave ""question"" y las preguntas dentro de una lista python"
        prompt_template = f"""
                      Puedes generar preguntas a partir del siguiente texto y generarlas con el siguiente formato diccionario de python?
                      formato de salida:
                      {output}

                      texto de entrada:
                      "{self.document}"
                      """

        # Generate questions
        return self.generator(prompt_template)

    def answer(self, row) -> str:
        # Create prompt for answer generation
        prompt_template = f"""
                      Bajo el siguiente contexto, puedes responder la siguiente pregunta:
                      "{row}"
                      Contexto:
                      "{self.document}"
                      """
        return self.generator(prompt_template)

    def __call__(self, document: str) -> pd.DataFrame:

        self.document = document

        # Generate questions:
        response = self.generate_questions()
        print("preguntas generadas")
        # Convertir la cadena de texto en un objeto JSON
        objeto_json = json.loads(response)

        # Create a DataFrame from the dictionary
        df_questions = pd.DataFrame(objeto_json)
        print("dataframe creado")
        # Answer questions
        df_questions["answer"] = df_questions.question.apply(self.answer)
        print("preguntas respondidas")
        return df_questions
