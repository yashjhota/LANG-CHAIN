from langchain.text_splitter import CharacterTextSplitter

text = """
    The quick brown fox jumps over the lazy dog. This is a sample text to demonstrate the functionality of the CharacterTextSplitter class. It is designed to split text into smaller chunks based on a specified character length. The quick brown fox jumps over the lazy dog. This is a sample text to demonstrate the functionality of the CharacterTextSplitter class. It is designed to split text into smaller chunks based on a specified character length.

"""

splitter = CharacterTextSplitter(
    chunk_size=50,  # Specify the desired chunk size
    chunk_overlap=0, # Specify the desired overlap between chunks
    separator=''  # Specify the separator to use for splitting
)

result = splitter.split_text(text)

print(result)