from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="play_tennis.csv"
)

docs = loader.load()

# it print row by row of the csv file

print(docs[0].page_content)