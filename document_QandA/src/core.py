from dotenv import load_dotenv, find_dotenv
from llama_index.callbacks import CallbackManager
from llama_index.chat_engine.types import ChatMode
from llama_index.llms import OpenAI
from llama_index.vector_stores import MilvusVectorStore
from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    LLMPredictor,
    OpenAIEmbedding,
    StorageContext,
    VectorStoreIndex
)
from phoenix.trace.llama_index import OpenInferenceTraceCallbackHandler
import phoenix as px

_ = load_dotenv(find_dotenv())

documents = SimpleDirectoryReader(input_dir="/Users/pavanmantha/Pavans/PracticeExamples/DataScience_Practice/LLMs/llama_index_tutorials/document_QandA/data", required_exts=[".pdf"]).load_data()
callbackHandler = OpenInferenceTraceCallbackHandler()

collection_name: str = 't55_product_specs'
db_uri: str = 'http://localhost:19530'


def load_defaults():
    service_context = ServiceContext.from_defaults(chunk_size=1000, chunk_overlap=200)
    # overwrite=True will override existing vectors in the store.
    vector_store = MilvusVectorStore(dim=1536, overwrite=False, collection_name=collection_name, uri=db_uri)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents=documents, service_context=service_context,
                                           storage_context=storage_context)


def load_gpt35_turbo():
    llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
    llm_predictor = LLMPredictor(llm=llm)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    # overwrite=True will override existing vectors in the store.
    vector_store = MilvusVectorStore(dim=1536, overwrite=False, collection_name=collection_name, uri=db_uri)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                                   embed_model=embed_model,
                                                   callback_manager=CallbackManager(handlers=[callbackHandler]),
                                                   chunk_size=500, chunk_overlap=100)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # save the documents index to vector store
    return VectorStoreIndex.from_documents(service_context=service_context, storage_context=storage_context,
                                           documents=documents)


# View the traces in the Phoenix UI
session = px.launch_app()
print(f'please launch your phoenix ui at {session.url}')

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


query_engine = load_gpt35_turbo().as_chat_engine(verbose=True, chat_mode=ChatMode.REACT)
result = query_engine.chat("What are the domains the Generative AI used in ?")
print(f'Bot: {result}')
#
# result = query_engine.chat("of those, what is his latest company?")
# print(f'Bot: {result}')
