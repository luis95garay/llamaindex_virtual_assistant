import logging
import pickle
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
load_dotenv("credentials.env")

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

# retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(),
#                                                   llm=ChatOpenAI(temperature=0))

# question = "Qué sabes de cardiología y cardiología Pediátrica?"
# question = "Qué especialidades tenemos disponibles?"
# question = "Puedes contarme un poco más sobre la especialidad de Fisioterapia que tenemos?"
question = "Puedes contarme cuál es la preparación para procedimiento de electrocardiograma?"

# docs = vectorstore.similarity_search(question)
# print(len(docs))

# unique_docs = retriever_from_llm.get_relevant_documents(query=question)
# print(len(unique_docs))

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Respuestas más grandes
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever()
    )

result = qa_chain({"query": question})
print(result)


# Respuestas más exactas
# qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
#     llm,
#     retriever=vectorstore.as_retriever()
#     )

# result = qa_chain({"question": question})
# print(result)




