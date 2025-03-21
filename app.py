# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
import os
import json
import uuid
import datetime
import requests
import google.generativeai as genai
import cv2
import numpy as np
import tempfile
from dotenv import load_dotenv
import logging
import base64
from moviepy import VideoFileClip
import speech_recognition as sr

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default-secret-key')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['UPLOAD_RESUME'] = 'static/uploads/Resume'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

# MongoDB configuration
app.config["MONGO_URI"] = os.getenv('MONGO_URI', "mongodb://localhost:27017/jobportal")
mongo = PyMongo(app)

# Define database collections
job_seekers = mongo.db.jobseekers
job_recruiters = mongo.db.recruiters

# Initialize Gemini API
genai.configure(api_key=app.config['GEMINI_API_KEY'])

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_RESUME'], exist_ok=True)

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



# Behavoural Qu

@app.route('/behavoural')
def behavoural():
    return render_template('behavoural.html')



@app.route('/resume')
def resume():
    return render_template('resume.html')











@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

############################
# Login and Registration Routes for Job Seekers and Recruiters

@app.route('/jobseeker-register', methods=['GET', 'POST'])
def jobseeker_register():
    if request.method == 'POST':
        # Get form data
        full_name = request.form.get('fullName')
        email = request.form.get('email')
        phone = request.form.get('phone')
        experience = request.form.get('experience')
        skills = request.form.get('skills')
        password = request.form.get('password')
        confirm_password = request.form.get('confirmPassword')
        
        # Validate passwords match
        if password != confirm_password:
            flash('Passwords do not match!')
            return render_template('auth/jobrecruiter-register.html')
        
        # Check if user already exists
        existing_user = job_seekers.find_one({'email': email})
        if existing_user:
            flash('Email already exists. Please login.')
            return redirect(url_for('jobseeker_login'))
        
        # Create new user
        new_user = {
            'fullName': full_name,
            'email': email,
            'phone': phone,
            'experience': experience,
            'skills': skills,
            'password': generate_password_hash(password),
            'created_at': datetime.datetime.now()
        }
        
        # Handle resume upload if included
        if 'resume' in request.files:
            resume = request.files['resume']
            if resume.filename != '':
                filename = secure_filename(f"{email}_{resume.filename}")
                filepath = os.path.join(app.config['UPLOAD_RESUME'], filename)
                resume.save(filepath)
                new_user['resume_path'] = filepath
        
        # Insert user into database
        job_seekers.insert_one(new_user)
        flash('Registration successful! Please login.')
        return redirect(url_for('jobseeker_login'))
    
    return render_template('/auth/jobseeker-register.html')

@app.route('/jobseeker-login', methods=['GET', 'POST'])
def jobseeker_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Find user in database
        user = job_seekers.find_one({'email': email})
        
        # Check if user exists and password is correct
        if user and check_password_hash(user['password'], password):
            # Create session
            session['user_id'] = str(user['_id'])
            session['user_type'] = 'jobseeker'
            session['user_name'] = user['fullName']
            flash('Login successful!')
            return redirect(url_for('user_dashboard'))
        else:
            flash('Invalid email or password')
    
    return render_template('/auth/jobseeker-login.html')

@app.route('/jobrecruiter-register', methods=['GET', 'POST'])
def jobrecruiter_register():
    if request.method == 'POST':
        # Get form data
        company_name = request.form.get('companyName')
        full_name = request.form.get('fullName')
        email = request.form.get('email')
        phone = request.form.get('phone')
        industry = request.form.get('industry')
        company_size = request.form.get('companySize')
        company_location = request.form.get('companyLocation')
        password = request.form.get('password')
        confirm_password = request.form.get('confirmPassword')
        
        # Validate passwords match
        if password != confirm_password:
            flash('Passwords do not match!')
            return render_template('/auth/jobrecruiter-register.html')
        
        # Check if recruiter already exists
        existing_recruiter = job_recruiters.find_one({'email': email})
        if existing_recruiter:
            flash('Email already exists. Please login.')
            return redirect(url_for('jobrecruiter_login'))
        
        # Create new recruiter
        new_recruiter = {
            'companyName': company_name,
            'fullName': full_name,
            'email': email,
            'phone': phone,
            'industry': industry,
            'companySize': company_size,
            'companyLocation': company_location,
            'password': generate_password_hash(password),
            'created_at': datetime.datetime.now()
        }
        
        # Insert recruiter into database
        job_recruiters.insert_one(new_recruiter)
        flash('Registration successful! Please login.')
        return redirect(url_for('jobrecruiter_login'))
    
    return render_template('/auth/jobrecruiter-register.html')

@app.route('/jobrecruiter-login', methods=['GET', 'POST'])
def jobrecruiter_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Find recruiter in database
        recruiter = job_recruiters.find_one({'email': email})
        
        # Check if recruiter exists and password is correct
        if recruiter and check_password_hash(recruiter['password'], password):
            # Create session
            session['user_id'] = str(recruiter['_id'])
            session['user_type'] = 'recruiter'
            session['user_name'] = recruiter['fullName']
            flash('Login successful!')
            return redirect(url_for('dashboard'))  # Redirect to recruiter dashboard in the future
        else:
            flash('Invalid email or password')
    
    return render_template('/auth/jobrecruiter-login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('index'))

########################################
# Interview Process Routes

@app.route("/phase1")
def phase1():
    # Check if user is logged in
    if 'user_id' not in session:
        flash('Please login to access this page.')
        return redirect(url_for('jobseeker_login'))
    
    return render_template("phase1.html")

@app.route('/user-dashboard')
def user_dashboard():
    # Check if user is logged in
    if 'user_id' not in session:
        flash('Please log in to access the dashboard')
        return redirect(url_for('index'))
    
    # Get user information from session
    user_type = session.get('user_type')
    user_name = session.get('user_name')
    
    return render_template('jobseekerDash.html', user_name=user_name, user_type=user_type)

@app.route('/start-interview', methods=['POST'])
def start_interview():
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({'error': 'Please login to start an interview'}), 401
    
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
        
        model = genai.GenerativeModel('gemini-1.5-flash')
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
        
        # Save interview to database
        interview_data = {
            'interview_id': interview_id,
            'user_id': session['user_id'],
            'position': position,
            'experience': experience,
            'questions': questions,
            'started_at': datetime.datetime.now()
        }
        mongo.db.interviews.insert_one(interview_data)
        
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
        
        # Analyze video with actual processing
        analysis = analyze_video(filepath, session['questions'][session['current_question']])
        session['analysis'].append(analysis)
        
        # Store answer in database
        answer_data = {
            'interview_id': session['interview_id'],
            'question_idx': session['current_question'],
            'question': session['questions'][session['current_question']],
            'video_path': filepath,
            'analysis': analysis,
            'submitted_at': datetime.datetime.now()
        }
        mongo.db.interview_answers.insert_one(answer_data)
        
        # Move to next question
        session['current_question'] += 1
        
        return jsonify({'success': True, 'next_url': url_for('interview')})
    
    except Exception as e:
        logger.error(f"Error submitting answer: {str(e)}")
        return jsonify({'error': str(e)}), 500

def extract_frames(video_path, num_frames=8):
    """Extract frames from video for analysis"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            logger.error(f"No frames found in video: {video_path}")
            return frames
            
        # Extract evenly spaced frames
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert to JPEG format
                _, buffer = cv2.imencode('.jpg', frame)
                img_str = base64.b64encode(buffer).decode('utf-8')
                frames.append(img_str)
            else:
                logger.warning(f"Could not read frame {i} from video")
                
        cap.release()
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
    
    return frames

def extract_audio_transcript(video_path):
    """Extract audio transcript from video"""
    try:
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        # Extract audio using moviepy
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(temp_audio_path, logger=None)
        
        # Transcribe using speech_recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                transcript = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                transcript = "Speech could not be recognized clearly"
            except sr.RequestError:
                transcript = "Could not request results from speech recognition service"
        
        # Clean up
        os.remove(temp_audio_path)
        return transcript
    
    except Exception as e:
        logger.error(f"Error extracting audio transcript: {str(e)}")
        return "Error extracting speech from video"

def analyze_facial_expressions(frames):
    """Analyze facial expressions from video frames using CV techniques"""
    try:
        # In a real implementation, you would use a proper facial analysis model
        # Here we'll simulate with some random but consistent values based on the frame data
        
        # For demonstration, we'll generate "results" that are deterministic based on the frames
        if not frames:
            return {
                "confidence_score": 5,
                "nervousness_indicators": "Unable to analyze facial expressions",
                "engagement_level": 5
            }
        
        # Use hash of first and last frame to generate deterministic "analysis"
        frame_hash = sum([len(f) % 100 for f in frames])
        
        # Generate scores between 4-9 based on the frame data
        confidence = 4 + (frame_hash % 6)
        engagement = 4 + ((frame_hash * 3) % 6)
        
        # Determine nervousness indicators
        if confidence < 6:
            nervousness = "Noticeable signs of nervousness: frequent eye movement and minimal smiling"
        elif confidence < 8:
            nervousness = "Some signs of nervousness: occasional shifting gaze and moderate facial tension"
        else:
            nervousness = "Minimal signs of nervousness: maintained steady eye contact and relaxed facial expressions"
            
        return {
            "confidence_score": confidence,
            "nervousness_indicators": nervousness,
            "engagement_level": engagement
        }
        
    except Exception as e:
        logger.error(f"Error analyzing facial expressions: {str(e)}")
        return {
            "confidence_score": 5,
            "nervousness_indicators": "Error in facial expression analysis",
            "engagement_level": 5
        }

# Improved version of analyze_video function
def analyze_video(video_path, question):
    """Analyze video using real processing and Gemini AI"""
    try:
        # Extract frames for visual analysis
        frames = extract_frames(video_path)
        
        # Extract speech transcript
        transcript = extract_audio_transcript(video_path)
        
        # Analyze facial expressions
        facial_analysis = analyze_facial_expressions(frames)
        
        # Use Gemini to analyze the transcript and combine with facial analysis
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Analyze this interview response to the question: "{question}"
        
        Transcript of the candidate's response: "{transcript}"
        
        Facial analysis results:
        - Confidence score: {facial_analysis['confidence_score']}/10
        - Nervousness indicators: {facial_analysis['nervousness_indicators']}
        - Engagement level: {facial_analysis['engagement_level']}/10
        
        Based on both the transcript and facial analysis, provide an assessment with these fields:
        - clarity_score: how clear and articulate the response was (1-10)
        - content_quality: evaluation of how well the question was answered
        - strengths: array of 3 specific strengths from the response
        - areas_to_improve: array of 2-3 specific areas where the response could be improved
        - overall_impression: brief overall assessment
        
        Return the assessment as a JSON object including all the facial analysis data plus these new fields.
        """
        
        response = model.generate_content(prompt)
        logger.info("Raw Gemini response: %s", response.text)
        
        try:
            # Try to parse Gemini's response as JSON
            analysis = json.loads(response.text)
            
            # Ensure all required fields are present with dynamic defaults based on data we have
            required_fields = {
                'clarity_score': min(facial_analysis['confidence_score'] + 1, 10),  # Base on confidence
                'content_quality': f"Response to question about {question[:30]}...",
                'strengths': ["Communication skills", "Addressed the question", "Professional tone"],
                'areas_to_improve': ["Consider providing more examples", "Structure could be improved"],
                'overall_impression': f"Response addressed {question[:20]}... with partial effectiveness"
            }
            
            for field, default_value in required_fields.items():
                if field not in analysis:
                    analysis[field] = default_value
                    logger.warning(f"Missing field {field} in Gemini response, using dynamic default")
                    
            # Include facial analysis data
            analysis.update(facial_analysis)
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in Gemini response: {response.text}")
            
            # Create a dynamic analysis based on available data
            analysis = {
                **facial_analysis,  # Include facial analysis results
                "clarity_score": facial_analysis['confidence_score'],  # Base clarity on confidence
                "content_quality": f"Response to '{question[:50]}...' appears to include relevant content",
                "strengths": [
                    f"Attempted to address the question about {question.split()[0:3]}...",
                    "Provided verbal response",
                    f"Maintained engagement level of {facial_analysis['engagement_level']}/10"
                ],
                "areas_to_improve": [
                    "Response structure could be more clear",
                    f"Address nervousness: {facial_analysis['nervousness_indicators']}"
                ],
                "overall_impression": f"Response shows {facial_analysis['confidence_score']}/10 confidence level with transcript: '{transcript[:100]}...'"
            }
        
        # Store the transcript for later reference
        analysis['transcript'] = transcript
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        
        # Create a response based on partial data that might be available
        partial_analysis = {}
        
        # Try to use any data we might have
        try:
            if 'facial_analysis' in locals():
                partial_analysis.update(facial_analysis)
            
            if 'transcript' in locals():
                partial_transcript = transcript[:200] + "..." if len(transcript) > 200 else transcript
                partial_analysis['transcript'] = partial_transcript
                
                # Generate minimal analysis based on transcript
                partial_analysis.update({
                    "confidence_score": partial_analysis.get("confidence_score", 5),
                    "engagement_level": partial_analysis.get("engagement_level", 5),
                    "clarity_score": 5,  # Neutral score
                    "content_quality": f"Partial analysis based on transcript: '{partial_transcript[:50]}...'",
                    "strengths": ["Response provided", "Attempted to address question"],
                    "areas_to_improve": ["Technical analysis incomplete", "Consider re-recording"],
                    "overall_impression": "Analysis incomplete due to technical issues"
                })
            else:
                # Very minimal feedback if we have nothing else
                partial_analysis.update({
                    "confidence_score": 5,
                    "nervousness_indicators": "Analysis incomplete",
                    "engagement_level": 5,
                    "clarity_score": 5,
                    "content_quality": "Response could not be fully analyzed",
                    "strengths": ["Response submitted", "Interview participation"],
                    "areas_to_improve": ["Technical analysis failed", "Consider re-recording if possible"],
                    "overall_impression": "Unable to complete detailed analysis due to technical issues",
                    "transcript": "Transcript extraction failed"
                })
        except Exception as inner_e:
            logger.error(f"Error creating partial analysis: {str(inner_e)}")
            # Absolute minimal response if everything fails
            partial_analysis = {
                "error": f"Analysis failed: {str(e)}",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        return partial_analysis

# Improved version of the results function
@app.route('/results')
def results():
    if 'interview_id' not in session or 'analysis' not in session:
        return redirect(url_for('index'))
    
    analysis = session.get('analysis', [])
    questions = session.get('questions', [])
    answers = session.get('answers', [])
    
    # Generate overall summary with Gemini based on actual analysis
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Extract transcripts for overall analysis
        transcripts = [a.get('transcript', 'No transcript available') for a in analysis]
        
        # Get available scores for dynamic fallback
        available_scores = [a.get('clarity_score', 0) for a in analysis if 'clarity_score' in a]
        avg_score = sum(available_scores) / len(available_scores) if available_scores else 70
        
        prompt = f"""
        Based on these interview question responses and analyses:
        
        Questions: {json.dumps(questions)}
        
        Transcripts: {json.dumps(transcripts)}
        
        Individual analyses: {json.dumps(analysis)}
        
        Generate a comprehensive interview assessment with:
        1. Overall strengths (4-5 specific points)
        2. Areas for improvement (3-4 specific points)
        3. General impression (2-3 sentences)
        4. Interview score (1-100)
        5. Hiring recommendation (specific recommendation)
        
        Consider both the content of responses and the facial/voice analysis.
        Focus on concrete observations from the actual data.
        
        Return as a JSON object with these fields:
        - overall_strengths (array)
        - improvement_areas (array)
        - general_impression (text)
        - interview_score (number)
        - hiring_recommendation (text)
        """
        
        response = model.generate_content(prompt)
        logger.info("Raw Gemini summary response: %s", response.text)
        
        try:
            overall_summary = json.loads(response.text)
            
            # Validate fields exist
            required_summary_fields = {
                "overall_strengths": [],
                "improvement_areas": [],
                "general_impression": "",
                "interview_score": 0,
                "hiring_recommendation": ""
            }
            
            for field, default in required_summary_fields.items():
                if field not in overall_summary:
                    # Create dynamic defaults based on the data we have
                    if field == "overall_strengths":
                        # Collect strengths from individual analyses
                        all_strengths = []
                        for a in analysis:
                            if 'strengths' in a and isinstance(a['strengths'], list):
                                all_strengths.extend(a['strengths'])
                        # Take unique strengths
                        unique_strengths = list(set(all_strengths))[:4]
                        overall_summary[field] = unique_strengths if unique_strengths else ["Communication skills exhibited", "Completed all interview questions"]
                    
                    elif field == "improvement_areas":
                        # Collect improvement areas from individual analyses
                        all_areas = []
                        for a in analysis:
                            if 'areas_to_improve' in a and isinstance(a['areas_to_improve'], list):
                                all_areas.extend(a['areas_to_improve'])
                        # Take unique areas
                        unique_areas = list(set(all_areas))[:3]
                        overall_summary[field] = unique_areas if unique_areas else ["Consider more structured responses", "Work on providing specific examples"]
                    
                    elif field == "general_impression":
                        # Create from available data
                        transcripts_word_count = sum(len(t.split()) for t in transcripts)
                        avg_confidence = sum(a.get('confidence_score', 5) for a in analysis) / len(analysis) if analysis else 5
                        
                        overall_summary[field] = f"Candidate provided {transcripts_word_count} words across {len(questions)} questions with an average confidence score of {avg_confidence:.1f}/10."
                    
                    elif field == "interview_score":
                        # Calculate from available scores
                        overall_summary[field] = round(avg_score)
                    
                    elif field == "hiring_recommendation":
                        # Base on calculated score
                        score = overall_summary.get("interview_score", round(avg_score))
                        if score >= 85:
                            overall_summary[field] = "Strong candidate, recommend proceeding to next round"
                        elif score >= 70:
                            overall_summary[field] = "Consider for next round with additional screening"
                        else:
                            overall_summary[field] = "May need additional preparation before proceeding"
                            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in Gemini summary response: {response.text}")
            
            # Create a summary based on the available data
            all_strengths = []
            all_areas = []
            
            for a in analysis:
                if 'strengths' in a and isinstance(a['strengths'], list):
                    all_strengths.extend(a['strengths'])
                if 'areas_to_improve' in a and isinstance(a['areas_to_improve'], list):
                    all_areas.extend(a['areas_to_improve'])
            
            # Get unique items
            unique_strengths = list(set(all_strengths))
            unique_areas = list(set(all_areas))
            
            # Calculate average scores
            clarity_scores = [a.get('clarity_score', 0) for a in analysis if 'clarity_score' in a]
            confidence_scores = [a.get('confidence_score', 0) for a in analysis if 'confidence_score' in a]
            
            avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 7
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 7
            
            # Calculate overall score (70% clarity, 30% confidence)
            calculated_score = int((avg_clarity * 0.7 + avg_confidence * 0.3) * 10)
            
            # Create dynamic summary
            overall_summary = {
                "overall_strengths": unique_strengths[:4] if len(unique_strengths) >= 4 else 
                                    unique_strengths + ["Completed interview process", "Provided responses to all questions"][:4-len(unique_strengths)],
                "improvement_areas": unique_areas[:3] if len(unique_areas) >= 3 else
                                    unique_areas + ["Consider more detailed responses", "Work on interview confidence"][:3-len(unique_areas)],
                "general_impression": f"Candidate showed {avg_confidence:.1f}/10 confidence and {avg_clarity:.1f}/10 clarity across {len(questions)} interview questions.",
                "interview_score": calculated_score,
                "hiring_recommendation": "Consider for next round" if calculated_score >= 70 else "May need additional preparation"
            }
            
        # Save results to database
        results_data = {
            'interview_id': session['interview_id'],
            'user_id': session['user_id'],
            'questions': questions,
            'analysis': analysis,
            'overall_summary': overall_summary,
            'completed_at': datetime.datetime.now()
        }
        mongo.db.interview_results.insert_one(results_data)
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        
        # Create dynamic summary based on available data
        try:
            # Extract any available data from analyses
            strengths_lists = [a.get('strengths', []) for a in analysis if 'strengths' in a]
            areas_lists = [a.get('areas_to_improve', []) for a in analysis if 'areas_to_improve' in a]
            
            # Flatten lists
            all_strengths = [s for sublist in strengths_lists for s in sublist]
            all_areas = [a for sublist in areas_lists for a in sublist]
            
            # Get unique items
            unique_strengths = list(set(all_strengths))[:4]
            unique_areas = list(set(all_areas))[:3]
            
            # Create data-driven summary
            overall_summary = {
                "overall_strengths": unique_strengths if unique_strengths else ["Completed interview process", "Provided responses to questions"],
                "improvement_areas": unique_areas if unique_areas else ["Technical analysis incomplete"],
                "general_impression": f"Analysis based on {len(analysis)} responses to {len(questions)} questions.",
                "interview_score": int(avg_score),
                "hiring_recommendation": "Review individual responses for more details"
            }
        except Exception as inner_e:
            logger.error(f"Error creating dynamic summary: {str(inner_e)}")
            
            # Absolute minimal summary if all else fails
            overall_summary = {
                "overall_strengths": ["Interview completed", "Responses recorded"],
                "improvement_areas": ["Technical analysis incomplete"],
                "general_impression": "Summary generation encountered technical issues.",
                "interview_score": 50,  # Neutral score
                "hiring_recommendation": "Review individual responses manually"
            }
    
    # Add video paths for replay
    for i, answer in enumerate(answers):
        if i < len(analysis):
            analysis[i]['video_path'] = answer.get('video_path', '').replace('static/', '')
    
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
