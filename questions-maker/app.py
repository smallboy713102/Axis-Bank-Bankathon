import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
import json
import os

openai_api_key = os.environ.get('OPENAI_API_KEY')
# Initialize your OpenAI language model here
llm = OpenAI(temperature=0.6, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k")

def generate_questionnaire(title, description, llm):

    question_template = """You are a member of the hiring committee of your company. Your task is to develop screening questions for each candidate, considering different levels of importance or significance assigned to the job description.
                                   Here are the Details:
                                   Job title: {title}
                                   Job description: {description}

                                Your Response should follow the following format:
                                 "id":1, "Question":"Your Question will go here"\n,
                                 "id":2, "Question":"Your Question will go here"\n,
                                 "id":3, "Question":"Your Question will go here"\n
                                 There should be at least 10 questions. Do output only the questions but in text."""
            
    screen_template = PromptTemplate(input_variables=["title", "description"], template=question_template)
    questions_chain = LLMChain(llm=llm, prompt=screen_template)
    
    response = questions_chain.run({"title": title, "description": description})
    
    return response

# Streamlit App
def main():
    st.title("Candidate Screening Questionnaire Generator")
    
    job_title = st.text_input("Enter Job Title:")
    job_description = st.text_area("Enter Job Description:")
    
    if st.button("Generate Questionnaire"):
        if job_title and job_description:
            questionnaire = generate_questionnaire(job_title, job_description, llm)
            
            st.write("Generated Questions:")
            st.write(questionnaire)
            
            question_strings = questionnaire.split('"id":')
            questions = []
            for q_string in question_strings[1:]:
                question_id, question_text = q_string.split(', "Question":')
                question = {
                    "id": int(question_id.strip()),
                    "Question": question_text.strip()[1:-1]  # Removing the surrounding quotes
                    }
                questions.append(question)
            questionnaire_json = json.dumps(questions, indent=4)
            
            # Make the questionnaire_json downloadable
            st.download_button(
                label="Download JSON Questionnaire",
                data=questionnaire_json,
                file_name="questionnaire.json",
                mime="application/json"
            )

        else:
            st.warning("Please provide Job Title and Job Description.")

if __name__ == "__main__":
    main()
