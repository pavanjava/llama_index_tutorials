import logging
import sys
import pandas as pd
from llama_index.query_engine import PandasQueryEngine
from IPython.display import Markdown, display

logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
logging.getLogger().addHandler(hdlr=logging.StreamHandler(stream=sys.stdout))

# Test on some sample data
df = pd.DataFrame(
    {
        "countries": ["India", "India", "India"],
        "city": ["Andhra Pradesh", "Tamil Nadu", "Karnataka"],
        "population": [2930000, 13960000, 3645000],
    }
)

# read it from external data source
# df = pd.read_csv("./data/norway_new_car_sales_by_make1.csv", encoding='unicode_escape')

query_engine = PandasQueryEngine(df=df, verbose=True)
response = query_engine.query("What is the city with the highest population?")
display(Markdown(f"<b>{response}</b>"))
