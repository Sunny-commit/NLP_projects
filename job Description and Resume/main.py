from flask import Flask, render_template, request
import os
import PyPDF2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    job_description = ""  # Placeholder for job description from form
    resume_text = ""  # Extracted resume text
    match_result = ""  # Placeholder for match result
    
    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        if 'resume' in request.files:
            file = request.files['resume']
            if file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                resume_text = extract_text_from_pdf(file_path)
                
                # Simple matching logic (can be replaced with NLP-based matching)
                if job_description.lower() in resume_text.lower():
                    match_result = "Resume matches the job description!"
                else:
                    match_result = "Resume does not match the job description."
    
    return render_template('matchresume.html', job_description=job_description, resume_text=resume_text, match_result=match_result)

if __name__ == '__main__':
    app.run(debug=True)
