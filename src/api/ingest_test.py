from langchain.document_loaders import BSHTMLLoader, UnstructuredHTMLLoader, WebBaseLoader, RecursiveUrlLoader, AsyncChromiumLoader
from dotenv import load_dotenv
import pickle
from langchain import OpenAI, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.chains import LLMChain, ConversationalRetrievalChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from bs4 import BeautifulSoup as Soup
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from src.data_processing.data_processing_pipeline import PipelineDataProcessing


load_dotenv("credentials.env")

pipe = PipelineDataProcessing(source_type="web")
pipe(
    path="https://mibluemedical.com/blue-medical-centro-de-especialidades/", 
    proceadure_type="add"
    )
# loader = RecursiveUrlLoader("https://mibluemedical.com/", extractor=lambda x: Soup(x, "html.parser").text)
# opcion 1
# loader = WebBaseLoader(["https://mibluemedical.com/blue-medical-centro-de-especialidades/"])
# html = loader.load()
# with open("output.txt", "w") as file:
#     file.write(html[0].page_content.replace("\n\n\n", ""))


# loader = CSVLoader(file_path='df_question.csv')
# data = loader.load()

# with open("output.txt", "w") as file:
#     file.write(data[0].page_content)

# print(len(data))
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
# )

# documents = text_splitter.split_documents(data)
# print(len(documents))

# with open("vectorstore.pkl", "rb") as f:
#         global vectorstore
#         vectorstore = pickle.load(f)

# question = ""
# docs = vectorstore.similarity_search(question)
# print(len(docs))
# embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.from_documents(documents, embeddings)

# Save vectorstore
# with open("vectorstore.pkl", "wb") as f:
#     pickle.dump(vectorstore, f)

# opcion 2
# Transform
# loader = AsyncChromiumLoader(["https://mibluemedical.com/blue-medical-centro-de-especialidades/"])
# html = loader.load()
# bs_transformer = BeautifulSoupTransformer()
# docs_transformed = bs_transformer.transform_documents(html, unwanted_tags=[]) 

# with open("output.txt", "w") as file:
#     file.write(html[0].page_content)

# question_prompt = PromptTemplate(input_variables=["text"], template=template)

# # LLM to generate questions
# llm = OpenAI()

# # # Chain to generate questions for a given text
# question_generator = LLMChain(llm=llm, prompt=question_prompt)

# # Generate questions
# questions = []
# # generated = question_generator.run(text=documents[0].page_content)

# # print(generated)
# for doc in raw_documents:
#   generated = question_generator.run(text=doc.page_content)
#   # questions.extend(generated.questions) 

# print(generated)
# print(type(generated))
# print(len(generated))



# Load documents
# with open("vectorstore.pkl", "rb") as f:
#     vectorstore = pickle.load(f)

# question_gen_llm = OpenAI(
#     temperature=0,
#     verbose=True,
# )
# streaming_llm = OpenAI(
#     streaming=True,
#     verbose=True,
#     temperature=0,
# )

# question_generator = LLMChain(
#     llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT
# )
# doc_chain = load_qa_chain(
#     streaming_llm, chain_type="stuff", prompt=QA_PROMPT
# )

# qa = ConversationalRetrievalChain(
#     retriever=vectorstore.as_retriever(),
#     combine_docs_chain=doc_chain,
#     question_generator=question_generator,
# )

# # qa = RetrievalQA.from_chain_type(streaming_llm,retriever=vectorstore.as_retriever(),
# #                                        chain_type="stuff")

# result = qa({"question": "puedes darme una lista de topicos mas importantes sobre blue medical", "chat_history": ""})
# print(result['answer'])