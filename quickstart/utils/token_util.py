from tiktoken import get_encoding, encoding_for_model
import logging
import json
import sys

tokenizer = get_encoding("cl100k_base")

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def count_tokens(text):
    tokenized_text = tokenizer.encode(text=text)
    tokenized_text = [{'token': token, 'text': tokenizer.decode([token])} for token in tokenized_text]
    logging.info(f"tokenized_text: {json.dumps(tokenized_text)}")
    return len(tokenized_text)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = encoding_for_model(encoding_name)
    encoded_str = encoding.encode(string)
    logging.info(f"encoded_text: {encoded_str}")
    num_tokens = len(encoded_str)
    return num_tokens


print(num_tokens_from_string("Hello world, let's test tiktoken.", "gpt-3.5-turbo"))

print(f'total tokens: {count_tokens("Hello world, lets test tiktoken.")}')
