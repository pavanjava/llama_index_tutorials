from dotenv import load_dotenv, find_dotenv
from llama_index.chat_engine.types import ChatMode
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

_ = load_dotenv(find_dotenv())

documents = SimpleDirectoryReader(input_dir="/Users/pavanmantha/Pavans/IMP-Docs", required_exts=[".pdf"]).load_data()


def load_defaults():
    service_context = ServiceContext.from_defaults(chunk_size=1000, chunk_overlap=200)
    return VectorStoreIndex.from_documents(documents=documents, service_context=service_context)


def load_gpt35_turbo():
    llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=500, chunk_overlap=100)
    return VectorStoreIndex.from_documents(documents=documents, service_context=service_context)


'''
persist the vector to a store on disk with a folder named 'storage', 
and we can use it from there at any point in time by loading the vectors from store back into our code as below.
If needed use the below code as per the need.
'''
# from llama_index import StorageContext, load_index_from_storage
#
# load_gpt35_turbo().storage_context.persist()
# storage_context = StorageContext.from_defaults(persist_dir="./storage")
# index = load_index_from_storage(storage_context=storage_context)

# query_engine = load_gpt35_turbo().as_query_engine()
# result = query_engine.query(str_or_query_bundle="What is the educational qualification of Pavan Kumar ?")
# print(result)

#
# query_engine = load_gpt35_turbo().as_chat_engine(verbose=False, chat_mode=ChatMode.REACT)
# result = query_engine.chat("What is the educational qualification of Pavan Kumar ?")
# print(f'Bot: {result}')
#
# result = query_engine.chat("of those, what is his latest company?")
# print(f'Bot: {result}')
