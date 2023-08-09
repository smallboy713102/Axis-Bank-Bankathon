import os
import streamlit as st
import csv
import json
import pandas as pd
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper


openai_api_key = os.environ.get('OPENAI_API_KEY')
zapier_nla_api_key = os.environ.get('ZAPIER_API_KEY')


# Set up Langchain components
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
zapier = ZapierNLAWrapper(zapier_nla_api_key=zapier_nla_api_key)
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(toolkit.get_tools(), llm, agent="zero-shot-react-description", verbose=True)

# Streamlit UI
st.title("HR's Shortlisted Candidates Email Sender")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    emails = df['Emails']
    
    json_data = []
    with uploaded_file as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            json_data.append(row)

    short = json.dumps(json_data)

    message = f""" Here are the emails of the candidates selected: {emails}. NO NEED to CC anyone. Make sure you do not include "[Candidate's Email]" in the "To" section.

    Your task is to send personalized congratulatory emails to the selected candidates, informing them about their selection and the next steps in the hiring process.

    Please craft individual emails for each candidate, addressing them by their name and including specific details about their selection and the next steps. Your emails should be professional, concise, and well-written, demonstrating enthusiasm for their selection and providing clear instructions on what they need to do next.

    Please note that each email should be unique and tailored to the individual candidate. You should avoid using any generic or template language. Instead, personalize each email by mentioning specific qualifications, experiences, or accomplishments that stood out during the selection process. Additionally, feel free to include any relevant information about the company, team, or role that may be of interest to the candidate.

    You may consult the following JSON object to gain specific information about each candidate: 

    {short}

    Ensure that the emails are error-free, have a professional tone, and are formatted correctly. Check the names and emails of the candidates to ensure accuracy before sending the emails.

    Your goal is to make each candidate feel appreciated, valued, and excited about the next steps in the hiring process.

    Make sure you have sent the emails to every eligible candidate selected.
    """

    st.text_area("Generated Message", message, height=300)

    if st.button("Send Emails"):
        agent.run(message)
        st.success("Emails sent successfully!")
        st.markdown("Click [here](https://huggingface.co/spaces/smallboy713102/Q-Maker) to visit the Q-Maker page.")
