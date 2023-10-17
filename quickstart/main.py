from llama_index import VectorStoreIndex, SimpleDirectoryReader
import logging
import sys
import openai
import warnings

warnings.filterwarnings("ignore")

openai.log = None
logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = SimpleDirectoryReader(input_dir="/Users/pavanmantha/Pavans/PracticeExamples/DataScience_Practice/LLMs"
                                            "/llama_index_tutorials/quickstart/data").load_data()
index = VectorStoreIndex.from_documents(documents=documents)

query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("what is author's experience on classification? ")
print(response.print_response_stream())
