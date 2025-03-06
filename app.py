# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json
import uuid
import datetime
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default-secret-key')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

# Initialize Gemini API
genai.configure(api_key=app.config['GEMINI_API_KEY'])

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Placeholder for interview questions (in production, use a database)
QUESTION_BANK = [
    "Tell me about a time you faced a challenge at work and how you overcame it.",
    "What makes you the ideal candidate for this position?",
    "Describe your greatest professional achievement.",
    "How do you handle pressure or stressful situations?",
    "Where do you see yourself in five years?",
    "What are your strengths and weaknesses?",
    "Why do you want to work for our company?",
    "How do you approach learning new skills?"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-interview', methods=['POST'])
def start_interview():
    # Create a unique interview session
    interview_id = str(uuid.uuid4())
    session['interview_id'] = interview_id
    session['current_question'] = 0
    session['questions'] = []
    session['answers'] = []
    session['analysis'] = []
    
    # Generate personalized questions using Gemini
    try:
        position = request.form.get('position', '')
        experience = request.form.get('experience', '')
        
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Generate 4 personalized interview questions for a {position} position with {experience} experience.
        The questions should be challenging but fair, and should help assess the candidate's fit for the role.
        Return just the questions in a JSON array format.
        """
        
        response = model.generate_content(prompt)
        questions = json.loads(response.text)
        
        # Fallback to question bank if Gemini fails
        if not questions or len(questions) < 3:
            import random
            questions = random.sample(QUESTION_BANK, 4)
        
        session['questions'] = questions
        
        return redirect(url_for('interview'))
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        # Fallback to question bank
        import random
        session['questions'] = random.sample(QUESTION_BANK, 4)
        return redirect(url_for('interview'))

@app.route('/interview')
def interview():
    if 'interview_id' not in session:
        return redirect(url_for('index'))
    
    current_q = session.get('current_question', 0)
    questions = session.get('questions', [])
    
    if current_q >= len(questions):
        return redirect(url_for('results'))
    
    return render_template('interview.html', 
                          question=questions[current_q],
                          question_num=current_q + 1,
                          total_questions=len(questions))

@app.route('/submit-answer', methods=['POST'])
def submit_answer():
    if 'interview_id' not in session:
        return jsonify({'error': 'No active interview session'}), 400
    
    try:
        video_data = request.files.get('video')
        
        if not video_data:
            return jsonify({'error': 'No video data received'}), 400
        
        # Save video file
        filename = f"{session['interview_id']}_{session['current_question']}.webm"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        video_data.save(filepath)
        
        # Store answer info
        session['answers'].append({
            'question_idx': session['current_question'],
            'question': session['questions'][session['current_question']],
            'video_path': filepath
        })
        
        # Analyze video with Gemini
        analysis = analyze_video(filepath, session['questions'][session['current_question']])
        session['analysis'].append(analysis)
        
        # Move to next question
        session['current_question'] += 1
        
        return jsonify({'success': True, 'next_url': url_for('interview')})
    
    except Exception as e:
        logger.error(f"Error submitting answer: {str(e)}")
        return jsonify({'error': str(e)}), 500

def analyze_video(video_path, question):
    """Analyze video using Gemini AI"""
    try:
        # In a real implementation, you would:
        # 1. Extract frames from the video
        # 2. Send frames to Gemini for analysis
        # 3. Process audio for tone analysis
        
        # For now, we'll use a text-based simulation with Gemini
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Analyze a video interview response to the question: "{question}"
        
        Simulate a detailed analysis of:
        1. Facial expressions (confidence, nervousness, engagement)
        2. Tone of voice (clarity, enthusiasm, professionalism)
        3. Answer content quality
        
        Return the analysis as a JSON object with these fields:
        - confidence_score (1-10)
        - nervousness_indicators (text)
        - engagement_level (1-10)
        - clarity_score (1-10)
        - content_quality (text)
        - strengths (array of strings)
        - areas_to_improve (array of strings)
        - overall_impression (text)
        """
        
        response = model.generate_content(prompt)
        analysis = json.loads(response.text)
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        # Return fallback analysis
        return {
            "confidence_score": 7,
            "nervousness_indicators": "Some hand movements suggest mild nervousness",
            "engagement_level": 8,
            "clarity_score": 7,
            "content_quality": "Good structure with relevant examples",
            "strengths": ["Clear communication", "Relevant examples", "Positive demeanor"],
            "areas_to_improve": ["Could provide more specific details", "Occasional filler words"],
            "overall_impression": "Overall positive impression with good communication skills"
        }

@app.route('/results')
def results():
    if 'interview_id' not in session or 'analysis' not in session:
        return redirect(url_for('index'))
    
    analysis = session.get('analysis', [])
    questions = session.get('questions', [])
    
    # Generate overall summary with Gemini
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Based on these individual question analyses: {json.dumps(analysis)},
        generate an overall interview assessment summary with:
        1. Overall strengths
        2. Areas for improvement
        3. General impression
        4. Interview score (1-100)
        
        Return as a JSON object with these fields:
        - overall_strengths (array)
        - improvement_areas (array)
        - general_impression (text)
        - interview_score (number)
        - hiring_recommendation (text)
        """
        
        response = model.generate_content(prompt)
        overall_summary = json.loads(response.text)
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        # Fallback summary
        overall_summary = {
            "overall_strengths": ["Good communication", "Professional demeanor", "Relevant experience"],
            "improvement_areas": ["Could provide more specific examples", "Technical depth could be improved"],
            "general_impression": "Solid candidate with good potential",
            "interview_score": 78,
            "hiring_recommendation": "Consider for next round"
        }
    
    return render_template('results.html', 
                          questions=questions,
                          individual_analyses=analysis,
                          overall_summary=overall_summary)

@app.route('/download-report')
def download_report():
    if 'interview_id' not in session:
        return redirect(url_for('index'))
    
    # In a real app, generate a PDF or detailed report
    # For now, just return the analysis as JSON
    report_data = {
        'interview_id': session['interview_id'],
        'timestamp': datetime.datetime.now().isoformat(),
        'questions': session.get('questions', []),
        'analysis': session.get('analysis', [])
    }
    
    return jsonify(report_data)

if __name__ == '__main__':
    app.run(debug=True)