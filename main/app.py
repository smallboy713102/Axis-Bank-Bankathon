import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import os
from io import BytesIO
import pickle
import pdfminer
from pdfminer.high_level import extract_text
import re
import PyPDF2
import textract
import tempfile
import pandas as pd
from docx import Document
import csv
import base64



nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

def extract_text_from_pdf(pdf_content):
    pdf_reader = PdfReader(BytesIO(pdf_content))
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_content):
    doc = Document(BytesIO(docx_content))
    text = " ".join(paragraph.text for paragraph in doc.paragraphs)
    return text


def extract_text_from_txt(txt_content):
    text = textract.process(input_filename=None, input_bytes=txt_content)
    return text

def extract_text_from_resume(file_path):
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == 'docx':
        return extract_text_from_docx(file_path)
    elif file_extension == 'txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def clean_pdf_text(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    text = re.sub('\s+', ' ', text)
    return text

def extract_candidate_name(text):
    pattern = r'(?:Mr\.|Ms\.|Mrs\.)?\s?([A-Z][a-z]+)\s([A-Z][a-z]+)'
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return "Candidate Name Not Found"

def calculate_similarity(job_description, cvs, cv_file_names):
    processed_job_desc = preprocess_text(job_description)

    processed_cvs = [preprocess_text(cv) for cv in cvs]

    all_text = [processed_job_desc] + processed_cvs

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)

    similarity_scores = cosine_similarity(tfidf_matrix)[0][1:]

    ranked_cvs = list(zip(cv_file_names, similarity_scores))
    ranked_cvs.sort(key=lambda x: x[1], reverse=True)

    return ranked_cvs

def extract_email_phone(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b(?:\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\d{3}[-.\s]??\d{4})\b'
    
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    
    return emails, phones



def rank_and_shortlist(job_description, cv_files, threshold=0.09):
    cv_texts = []
    cv_file_names = []
    cv_emails = []
    cv_phones = []

    for cv_file in cv_files:
        file_extension = os.path.splitext(cv_file.name)[1].lower()

        try:
            if file_extension == '.pdf':
                cv_text = extract_text_from_pdf(cv_file.read())
            elif file_extension == '.docx':
                cv_text = extract_text_from_docx(cv_file.read())
            elif file_extension == '.txt':
                cv_text = cv_file.read().decode('utf-8', errors='ignore')
            else:
                st.warning(f"Unsupported file format: {file_extension}. Skipping file: {cv_file.name}")
                continue

            cv_texts.append(clean_pdf_text(cv_text))
            cv_file_names.append(cv_file.name)

            # Extract email and phone number from the CV text
            emails, phones = extract_email_phone(cv_text)
            cv_emails.append(emails)
            cv_phones.append(phones)

        except Exception as e:
            st.warning(f"Error processing file '{cv_file.name}': {str(e)}")
            continue

    if not cv_texts:
        st.error("No valid resumes found. Please upload resumes in supported formats (PDF, DOCX, or TXT).")
        return [], {}

    similarity_scores = calculate_similarity(job_description, cv_texts, cv_file_names)

    ranked_cvs = [(cv_name, score) for (cv_name, score) in similarity_scores]
    shortlisted_cvs = [(cv_name, score) for (cv_name, score) in ranked_cvs if score >= threshold]

    
    contact_info_dict = {}
    for cv_name, emails, phones in zip(cv_file_names, cv_emails, cv_phones):
        contact_info_dict[cv_name] = {
            'emails': emails,
            'phones': phones,
        }

    return ranked_cvs, shortlisted_cvs, contact_info_dict

def export_to_csv(data, filename):
    df = pd.DataFrame(data.items(), columns=['File Name', 'Emails'])
    df.to_csv(filename, index=False)


def main():
    st.title("HireGPT")

    st.write("Enter Job Title:")
    job_title = st.text_input("Job Title")

    st.write("Enter Job Description:")
    job_description = st.text_area("Job Description", height=200, key='job_description')

    st.markdown('[![Enhance Job Description](https://img.shields.io/badge/Enhance_Job_Description-Click_Here-brightgreen)](https://huggingface.co/spaces/smallboy713102/Enhancer)')


    st.write("Upload the Resumes:")
    cv_files = st.file_uploader("Choose files", accept_multiple_files=True, key='cv_files')

    if st.button("Submit"):
        if job_title and job_description and cv_files:
            job_description_text = f"{job_title} {job_description}"

            ranked_cvs, shortlisted_cvs, contact_info_dict = rank_and_shortlist(job_description_text, cv_files)

            st.markdown("### Ranking of Resumes:")
            for rank, score in ranked_cvs:
                st.markdown(f"**File Name:** {rank}, **Similarity Score:** {score:.2f}")

            st.markdown("### Shortlisted Candidates:")
            if not shortlisted_cvs:
                st.markdown("None")
            else:
                shortlisted_candidates_data = {}
                for rank, score in shortlisted_cvs:
                    st.markdown(f"**File Name:** {rank}, **Similarity Score:** {score:.2f}")

                    contact_info = contact_info_dict[rank]
                    candidate_emails = contact_info.get('emails', [])
                    if candidate_emails:
                        shortlisted_candidates_data[rank] = candidate_emails
                        st.markdown(f"**Emails:** {', '.join(candidate_emails)}")

                if shortlisted_candidates_data:
                    export_filename = "shortlisted_candidates.csv"
                    temp_dir = tempfile.gettempdir()
                    temp_file_path = os.path.join(temp_dir, export_filename)
                    export_to_csv(shortlisted_candidates_data, temp_file_path)
                    with open(temp_file_path, 'rb') as file:
                        csv_content = file.read()
                        b64_encoded_csv = base64.b64encode(csv_content).decode()
                    st.markdown(
                        f'<a href="data:application/octet-stream;base64,{b64_encoded_csv}" download="{export_filename}">'
                        '<button style="padding: 10px; background-color: #4CAF50; color: white; border: none; cursor: pointer;">'
                        'Download CSV</button></a>',unsafe_allow_html=True
                        )
    
                    st.markdown(
                        '<a href="https://huggingface.co/spaces/smallboy713102/Shortlisted_Candidate_Email_Sender" '
                        'target="_blank"><button style="padding: 10px; background-color: #008CBA; color: white; border: none; cursor: pointer;">'
                        'HR\'s Shortlisted Email Sender</button></a>',unsafe_allow_html=True
                        )

        else:
            st.error("Please enter the job title, job description, and upload resumes to proceed.")
    else:
        st.write("Please enter the job title, job description, and upload resumes to proceed.")

if __name__ == "__main__":
    main()
