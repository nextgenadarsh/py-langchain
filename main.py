from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from dotenv import load_dotenv

from src.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    print("Welcome to Langchain !!")

    load_dotenv()

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    # llm = ChatOllama(temperature=0, model="llama3")
    llm = ChatOllama(temperature=0, model="mistral")
    chain = summary_prompt_template | llm | StrOutputParser()
    linkedin_data = scrape_linkedin_profile(
        "http://linkedin.com/in/nextgenadarsh/", True
    )
    res = chain.invoke(input={"information": linkedin_data})
    print(res)
