import streamlit as st
import os
import vertexai
import wikipedia 
import nltk
from nltk.corpus import stopwords
from collections import Counter
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from rag import *
import wikipediaapi
from langchain_groq import ChatGroq

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./key.json"
os.environ["GROQ_API_KEY"] = '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'

PROJECT_ID = "nih-cl-cm500-chenhonl-3a5d" 
REGION = "us-central1"

vertexai.init(project=PROJECT_ID, location=REGION)

# Gemini Version
llm = VertexAI(
    model_name="gemini-1.5-pro-001",
    max_output_tokens=512,
    temperature=0.5,
    top_p=0.9,
    top_k=40,
    verbose=True,
)

#llm = ChatGroq(
#        model_name = "llama-3.3-70b-versatile",
#        temperature= 0.1,
#        verbose=True,
#    )


st.title("AI-Powered Code Generation Assistant")
languages = ["Python", "Java", "Matlab", "C++", "C"]
domains = ["Machine Learning", "Web Development", "Data Engineering", "Cybersecurity", "Embedded Systems",
           "Cloud Computing"]
selected_language = st.selectbox("Select a Programming Language:", languages)
selected_domain = st.selectbox("Select a Programming Domain:", domains)
st.write(f"You selected **{selected_language}** for **{selected_domain}**.")

st.subheader("AI Code Generator")
user_query = st.text_input("Describe what you need (e.g., 'Create a simple web server'):")


if user_query:
    with st.spinner("Generating code..."):
        prompt_template = PromptTemplate(
            input_variables=['language', 'domain', 'query'],
            template=(
                "Generate a {language} code snippet related to {domain}."
                " Task description: {query}"
                " Provide well-structured and commented code."
            )
        )

        chain = LLMChain(llm=llm, prompt=prompt_template)
        result = chain.invoke({"language": selected_language, "domain": selected_domain, "query": user_query})
        generated_code = result['text'].strip()
        st.code(generated_code, language=selected_language.lower())

        
    #####
    # RAG
    ####
    st.subheader("Retrieval Augmented Generation + LLM")
    output, _ , meta_data = retrieval_augmented_generation(user_query+selected_language)
    prompt_template = PromptTemplate(
            input_variables=['code_snippet', "meta_data", "language", "query"],
            template=(
            '''You are a helpful and professional AI coding assistant for programmers. 
                 Read this code_snippet {code_snippet} and choose one snipped that is most relevant to
                 {language} and {query} from user. Explain how it is working to the user in 4
                 sentences. Make sure to read the {meta_data} and explain the metadata as it relates to
                 the code snippet and provide the url. 
                 Follow this template and do not digress from this:
                 1. Code snippet in pretty print
                 2. Explanation
                 3. Metadata explanation
                 4. Additional explanation if {code_snippet} doesn't address the {query}
                 5. Provide documentation link for {language}
                 6. Provide documentation link for that might we helpful'''
            )
        )    
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.invoke({"code_snippet": output, 
                           "meta_data": meta_data, 
                           "language": selected_language,
                           "query": user_query})
    st.write(result['text'].strip())
    lines = output[0].page_content.split("\n")
    joined_string = "\n".join(lines)
    #st.code(joined_string, language=selected_language.lower())

    #####
    # Wikipedia RAG
    #####
    st.subheader("Wikipedia agent with LLM and RAG")
    wiki_wiki = wikipediaapi.Wikipedia(user_agent="project")
    page = wiki_wiki.page(selected_language+" programming") #python programming
    wiki_links = list(page.links.keys())

    prompt_template_title = PromptTemplate(
            input_variables=["language", "wikipedia_links"],
            template=(
            '''You are a helpful and professional AI assistant that looks at {wikipedia_links} and 
            chooses the most related 4 wikipedia titles to the programming {language}. 
            You will present these three titles as python string like so: title1, title2, title3, title4.
            Do not provide anything else.'''
            )
        )    
    
    chain = LLMChain(llm=llm, prompt=prompt_template_title)
    result = chain.invoke({"language": selected_language, "wikipedia_links": wiki_links})
    

    prompt_template = PromptTemplate(
            input_variables=["wikipedia_summary"],
            template=(
            '''You are a helpful and professional AI assistant that looks at {wikipedia_summary} and 
            provides 4 sentence summary. Do not provide anything else.'''
            )
        )    

    prompt_template_domain = PromptTemplate(
            input_variables=["language", "wikipedia_links"],
            template=(
            '''You are a helpful and professional AI assistant that looks at {wikipedia_links} and 
            chooses the most related 4 wikipedia titles to the programming {language} and {domain}. 
            You will present these three titles as python string like so: title1, title2, title3, title4.
            Do not provide anything else.'''
            )
        )    

    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    splitted = result["text"].split(",")
    for idx in range(1, len(splitted)):
        ignore_first = splitted[idx]
        cleaned_text = ignore_first.replace("'", "").strip()
        cleaned_text = cleaned_text.replace('"', "").strip()
        page = wiki_wiki.page(cleaned_text)
        st.write(f"{page.title}")
        result = chain.invoke({"wikipedia_summary": page.summary})
        st.write(result["text"].strip()) # 4 sentence summary for wikipedia art
        st.write(f"Please read more here : {page.fullurl}")
    
    page = wiki_wiki.page(selected_domain) #machine learning
    wiki_links = list(page.links.keys())
    chain = LLMChain(llm=llm, prompt=prompt_template_domain)
    result = chain.invoke({"language":selected_language,  "domain": selected_domain, "wikipedia_links": wiki_links})

   
    splitted = result["text"].split(",")
    chain = LLMChain(llm=llm, prompt=prompt_template)
    for idx in range(1, len(splitted)):
        ignore_first = splitted[idx]
        cleaned_text = ignore_first.replace("'", "").strip()
        cleaned_text = cleaned_text.replace('"', "").strip()
        page = wiki_wiki.page(cleaned_text)
        st.write(f"{page.title}")
        result = chain.invoke({"wikipedia_summary": page.summary})
        st.write(result["text"].strip())
        st.write(f"Please read more here : {page.fullurl}")


