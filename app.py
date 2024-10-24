from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import PyPDF2
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdf_file.stream as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()  # Remove any leading/trailing whitespace

@app.route('/analyze_cv', methods=['POST'])
def analyze_cv():
    cv_file = request.files['cv']
    job_description = request.form['job_description']

    # Extract text from the PDF CV
    cv_text = extract_text_from_pdf(cv_file)

    # Call the Gemini API to analyze the CV
    analysis_result = call_gemini_api(cv_text, job_description)

    # Extract relevant analysis content from the response
    if 'candidates' in analysis_result and len(analysis_result['candidates']) > 0:
        analysis_content = analysis_result['candidates'][0]['content']['parts'][0]['text']
    else:
        analysis_content = "No analysis content available."

    # Return the analysis result as JSON with proper decoration
    response = {
        "analysis": analysis_content,
        "cv_text": cv_text,
        "job_description": job_description
    }

    return jsonify(response)

def call_gemini_api(cv_text, job_description):
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

    # Prepare the payload for the API request with the new prompt
    prompt = (
        f"Hey, act like a skilled or very experienced ATS (Application Tracking System) with a deep understanding of tech field of {job_description}."
        f"Your task is to evaluate the resume based on the given job description. "
        f"You should provide the best assistance for improving their resumes. "
        f"Assign the percentage matching based on {job_description} and the missing keywords with high accuracy."
        f"Be honest with the score, even if the score gets 0% match."
        f"Also suggest a better alternative job role based on my technical skills.\n\n"
        f"CV Text:\n{cv_text}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    # Make the API request
    response = requests.post(f"{gemini_api_url}?key={gemini_api_key}", json=payload, headers=headers)

    # Check for successful response
    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": "Failed to analyze CV",
            "status_code": response.status_code,
            "details": response.json()  # Include response details for debugging
        }

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
