from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load LCEL docs
url = "https://python.langchain.com/docs/expression_language/"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# Sort the list based on the URLs in 'metadata' -> 'source'
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))

# Concatenate the 'page_content' of each sorted dictionary
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)

# Prompt template
template = """You are a coding assistant with expertise in LCEL, LangChain expression language. Here is a full set of documentation:
{context}

Now, answer the user question based on the above provided documentation: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatAnthropic(temperature=0, model="claude-2.1")

chain = (
    {
        "question": RunnablePassthrough(),
    }
    | prompt.partial(context=concatenated_content)
    | model
    | StrOutputParser()
)
