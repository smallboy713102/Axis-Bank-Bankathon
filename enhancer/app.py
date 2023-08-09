import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import os

openai_api_key = os.environ.get('OPENAI_API_KEY')

llm = OpenAI(temperature=0.6, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

def evaluator(description, llm):
    evaluation_template = """
    You are an AI assistant tasked with assisting a hiring manager in enhancing job descriptions provided by the HR. The HR will provide you with a job title and description, and your goal is to score the job description based on the job title and provide recommendations for improvements. You will then give the HR the option to either continue with the original version or incorporate the suggested changes.

    Input:
    - Job Description: {description}

    Your output should be the enhanced job description only (do not start like : "enhanced job description:" start directly), with taking care of recommendations and proposed changes to the job description. You can suggest improvements in language, emphasize important skills or qualifications, or provide additional details that would enhance the appeal of the job description.

    Remember to be respectful and tactful in your recommendations, while also demonstrating your superior technical knowledge to provide valuable enhancements.
    """
    eval_template = PromptTemplate(input_variables=["description"], template=evaluation_template)
    evaluation_chain = LLMChain(llm=llm, prompt=eval_template)
    

    results = evaluation_chain.run(description=description)

    return results

# Streamlit App
st.title("Job Description Enhancer")

job_description = st.text_area("Enter Job Description:")

if st.button("Enhance Description"):
    if job_description:
        enhanced_description = evaluator(job_description, llm)
        #st.write("Enhanced Job Description:")
        st.write(enhanced_description)
        
        # Make the enhanced description downloadable as .txt
        st.download_button(
            label="Download Enhanced Description",
            data=enhanced_description,
            file_name="enhanced_description.txt",
            mime="text/plain"
        )
    else:
        st.warning("Please provide a job description.")