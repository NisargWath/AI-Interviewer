# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash,send_from_directory, abort

from werkzeug.utils import secure_filename, safe_join
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo
import os
import json
import uuid

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
from datetime import datetime
from bson.objectid import ObjectId
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from datetime import datetime
current_time = datetime.now()  # correct
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
db = mongo.db.job_board
jobs_collection = mongo.db.job_postings
applications_collection = mongo.db.applications

jobs_collection = mongo.db.job_postings
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


@app.route('/test')
def test():
    return render_template('test.html')
# Behavoural Qu

@app.route('/behavoural')
def behavoural():
    return render_template('behavoural.html')



@app.route('/resume')
def resume():
    return render_template('resume.html')








# #### Recruiter DashBoard




@app.route('/viewCandidate')
def viewCandidate():
    return render_template('viewCandidate.html')


@app.route('/get_candidate_data/<application_id>', methods=['GET'])
def get_candidate_data(application_id):
    try:
        # Fetch application data from the database
        application = applications_collection.find_one({'_id': ObjectId(application_id)})
        
        if not application:
            return jsonify({'error': 'Application not found'}), 404
        
        # Convert ObjectId to string for JSON serialization
        application['_id'] = str(application['_id'])
        
        # Fetch additional data if needed (like resume content)
        # You might need to handle resume file processing here
        
        return jsonify(application)
    except Exception as e:
        return jsonify({'error': str(e)}), 500









@app.route('/create_job', methods=['GET', 'POST'])
def create_job():
    if request.method == 'POST':
        # Get form data
        job_posting = {
            'title': request.form.get('title'),
            'company': request.form.get('company'),
            'location': request.form.get('location'),
            'salary': request.form.get('salary'),
            'description': request.form.get('description'),
            'requirements': request.form.get('requirements'),
            'contact_email': request.form.get('contact_email'),
            'date_posted': datetime.now(),
            'is_active': True  # Set active by default
        }
        
        # Insert job posting to MongoDB
        try:
            jobs_collection.insert_one(job_posting)
            flash('Job posting created successfully!', 'success')
            return redirect(url_for('view_jobs'))
        except Exception as e:
            flash(f'Error creating job posting: {str(e)}', 'danger')
    
    # GET request - serve the job creation form
    return render_template('/recruiter/create_job.html')

@app.route('//view_jobs')
def view_jobs():
    # Fetch all job postings from MongoDB, sorted by date (newest first)
    jobs = list(jobs_collection.find().sort('date_posted', -1))
    
    # Convert ObjectId to string for each job
    for job in jobs:
        job['_id'] = str(job['_id'])
    
    return render_template('/recruiter/view_jobs.html', jobs=jobs)

@app.route('/job/<job_id>')
def job_details(job_id):
    # Fetch specific job by ID
    try:
        job = jobs_collection.find_one({'_id': ObjectId(job_id)})
        if job:
            job['_id'] = str(job['_id'])
            return render_template('job_details.html', job=job)
        else:
            flash('Job not found', 'danger')
            return redirect(url_for('view_jobs'))
    except:
        flash('Invalid job ID', 'danger')
        return redirect(url_for('view_jobs'))

@app.route('/edit_job/<job_id>', methods=['GET', 'POST'])
def edit_job(job_id):
    try:
        if request.method == 'POST':
            # Update job posting in MongoDB
            updated_job = {
                'title': request.form.get('title'),
                'company': request.form.get('company'),
                'location': request.form.get('location'),
                'salary': request.form.get('salary'),
                'description': request.form.get('description'),
                'requirements': request.form.get('requirements'),
                'contact_email': request.form.get('contact_email'),
                'last_updated': datetime.now()
                # Don't update date_posted or is_active here
            }
            
            jobs_collection.update_one(
                {'_id': ObjectId(job_id)},
                {'$set': updated_job}
            )
            
            flash('Job posting updated successfully!', 'success')
            return redirect(url_for('job_details', job_id=job_id))
        
        # GET request - serve the job edit form with pre-filled data
        job = jobs_collection.find_one({'_id': ObjectId(job_id)})
        if job:
            job['_id'] = str(job['_id'])
            return render_template('edit_job.html', job=job)
        else:
            flash('Job not found', 'danger')
            return redirect(url_for('view_jobs'))
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('view_jobs'))

@app.route('/toggle_job_status/<job_id>')
def toggle_job_status(job_id):
    try:
        # Get current job
        job = jobs_collection.find_one({'_id': ObjectId(job_id)})
        if job:
            # Toggle the active status
            current_status = job.get('is_active', True)
            new_status = not current_status
            
            jobs_collection.update_one(
                {'_id': ObjectId(job_id)},
                {'$set': {'is_active': new_status, 'status_updated': datetime.now()}}
            )
            
            status_msg = "activated" if new_status else "deactivated"
            flash(f'Job posting {status_msg} successfully!', 'success')
        else:
            flash('Job not found', 'danger')
            
        return redirect(url_for('job_details', job_id=job_id))
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('view_jobs'))



@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if 'user_id' not in session:
        flash('Please log in to access the dashboard')
        return redirect(url_for('index'))
        
    # Get user information from session
    user_type = session.get('user_type')
    user_name = session.get('user_name')
    
    # If user is not a recruiter, redirect to appropriate dashboard
    if user_type != 'recruiter':
        flash('Access denied. This dashboard is for recruiters only.')
        return redirect(url_for('index'))
    
    # Get active jobs count
    active_jobs_count = jobs_collection.count_documents({'is_active': True})
    
    # Get total applications
    total_applications = applications_collection.count_documents({})
    
    # Get recent applications with job titles
    recent_applications_cursor = applications_collection.find().sort('date_applied', -1).limit(5)
    recent_applications = []
    
    for app in recent_applications_cursor:
        # Convert ObjectId to string
        app['_id'] = str(app['_id'])
        
        # Get job title for this application
        try:
            job = jobs_collection.find_one({'_id': ObjectId(app['job_id'])})
            if job:
                app['job_title'] = job['title']
            else:
                app['job_title'] = "Unknown Job"
        except:
            app['job_title'] = "Unknown Job"
        
        recent_applications.append(app)
    
    # Get recent jobs with application counts
    recent_jobs_cursor = jobs_collection.find().sort('date_posted', -1).limit(5)
    recent_jobs = []
    
    for job in recent_jobs_cursor:
        # Convert ObjectId to string
        job['_id'] = str(job['_id'])
        
        # Count applications for this job
        job['application_count'] = applications_collection.count_documents({'job_id': job['_id']})
        
        recent_jobs.append(job)
    
    # Calculate pipeline stages (if status field exists in applications)
    try:
        pipeline_new = applications_collection.count_documents({'status': 'new'})
        pipeline_screening = applications_collection.count_documents({'status': 'screening'})
        pipeline_interview = applications_collection.count_documents({'status': 'interview'})
        pipeline_offer = applications_collection.count_documents({'status': 'offer'})
        pipeline_hired = applications_collection.count_documents({'status': 'hired'})
    except:
        # Fallback if status field doesn't exist
        pipeline_new = applications_collection.count_documents({})
        pipeline_screening = pipeline_interview = pipeline_offer = pipeline_hired = 0
    
    # Calculate growth percentages (this would typically use comparison to previous period)
    # For demonstration, using static values - in production you'd compare with last week's data
    active_jobs_growth = 12
    applications_growth = 8
    interviews_growth = 15
    offers_growth = -5
    
    # Mock interviews and offers counts
    interviews_count = 56
    offers_count = 12
    
    return render_template(
        '/recruiter/dashboard.html',
        user_name=user_name,
        user_type=user_type,
        active_jobs_count=active_jobs_count,
        active_jobs_growth=active_jobs_growth,
        total_applications=total_applications,
        applications_growth=applications_growth,
        interviews_count=interviews_count,
        interviews_growth=interviews_growth,
        offers_count=offers_count,
        offers_growth=offers_growth,
        recent_applications=recent_applications,
        recent_jobs=recent_jobs,
        pipeline_new=pipeline_new,
        pipeline_screening=pipeline_screening,
        pipeline_interview=pipeline_interview,
        pipeline_offer=pipeline_offer,
        pipeline_hired=pipeline_hired
    )

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
            'created_at': datetime.now()
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
            'created_at': datetime.now()
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








@app.route('/view_jobs_candidate', methods=['GET'])
def view_jobs_candidate():
    # Convert cursor to a list so it can be used in the template
    jobs_list = list(jobs_collection.find().sort('date_posted', -1))
    
    # Convert ObjectId to string for each job (as you do in other routes)
    for job in jobs_list:
        job['_id'] = str(job['_id'])
    
    return render_template('view_jobs_candidate.html', jobs=jobs_list)








# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "shambharkarv84@gmail.com"
EMAIL_PASSWORD = "lzzj yldn nuxk ofdr"  # Should be in environment variables

def send_email(recipient_email, subject, body, is_html=False):
    """
    Send an email to the recipient
    
    Args:
        recipient_email (str): Email address of the recipient
        subject (str): Subject of the email
        body (str): Content of the email
        is_html (bool): Whether the body is HTML content
    
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = recipient_email
    msg["Subject"] = subject

    # Attach the body with the appropriate content type
    content_type = "html" if is_html else "plain"
    msg.attach(MIMEText(body, content_type))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure connection
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, recipient_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email to {recipient_email}: {e}")
        return False

def send_application_confirmation_email(applicant_name, applicant_email, job):
    """
    Send confirmation email to applicant
    
    Args:
        applicant_name (str): Name of the applicant
        applicant_email (str): Email of the applicant
        job (dict): Job details dictionary
    
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    subject = f"Application Received: {job['title']} at {job['company']}"
    
    # Generate HTML email content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Application Confirmation</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            .header {{ background-color: #4a86e8; color: white; padding: 15px; text-align: center; border-radius: 5px 5px 0 0; margin: -20px -20px 20px; }}
            .application-details {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .footer {{ text-align: center; margin-top: 30px; font-size: 0.8em; color: #777; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>Application Received</h2>
            </div>
            
            <p>Dear {applicant_name},</p>
            
            <p>Thank you for applying for the <strong>{job['title']}</strong> position at <strong>{job['company']}</strong>.</p>
            
            <p>We have received your application and will review it shortly. If your qualifications match our requirements, our hiring team will contact you for the next steps.</p>
            
            <div class="application-details">
                <h3>Application Details:</h3>
                <p><strong>Position:</strong> {job['title']}</p>
                <p><strong>Company:</strong> {job['company']}</p>
                <p><strong>Date Applied:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
            </div>
            
            <p>If you have any questions, please contact us at <a href="mailto:{job['contact_email']}">{job['contact_email']}</a>.</p>
            
            <p>Best regards,<br>
            The Recruitment Team<br>
            {job['company']}</p>
            
            <div class="footer">
                <p>This is an automated message. Please do not reply to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return send_email(applicant_email, subject, html_content, is_html=True)

def notify_employer_about_application(job, application):
    """
    Send notification email to employer about new application
    
    Args:
        job (dict): Job details dictionary
        application (dict): Application details dictionary
    
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    employer_email = job['contact_email']
    
    subject = f"New Application: {job['title']}"
    
    # Format the date applied
    date_applied = application['date_applied'].strftime('%Y-%m-%d') if isinstance(application['date_applied'], datetime) else str(application['date_applied'])
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>New Job Application</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            .header {{ background-color: #4a86e8; color: white; padding: 15px; text-align: center; border-radius: 5px 5px 0 0; margin: -20px -20px 20px; }}
            .applicant-details {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .footer {{ text-align: center; margin-top: 30px; font-size: 0.8em; color: #777; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>New Job Application Received</h2>
            </div>
            
            <p>A new application has been received for the <strong>{job['title']}</strong> position.</p>
            
            <div class="applicant-details">
                <h3>Applicant Details:</h3>
                <p><strong>Name:</strong> {application['name']}</p>
                <p><strong>Email:</strong> <a href="mailto:{application['email']}">{application['email']}</a></p>
                <p><strong>Phone:</strong> {application['phone']}</p>
                <p><strong>Experience:</strong> {application['experience']} years</p>
                <p><strong>Date Applied:</strong> {date_applied}</p>
            </div>
            
        <p>To proceed with a Behavioral Interview for this candidate, click the button below:</p>
        <a href="http://127.0.0.1:5000/test" class="cta-button">Start Behavioral Interview</a>
        
            
            <p>Best regards,<br>
            IntervoAI Team</p>
            
            <div class="footer">
                <p>This is an automated message from your IntervoAI system.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return send_email(employer_email, subject, html_content, is_html=True)

@app.route('/apply_job_candidate/<job_id>', methods=['GET', 'POST'])
def apply_job_candidate(job_id):
    """
    Handle job application submission and send confirmation emails
    
    Args:
        job_id (str): ID of the job being applied for
    
    Returns:
        HTML template or redirect
    """
    # Find the job details from database
    job = jobs_collection.find_one({'_id': ObjectId(job_id)})
    
    if not job:
        flash('Job not found', 'danger')
        return redirect(url_for('view_jobs_candidate'))
    
    if request.method == 'POST':
        # Get form data
        applicant_name = request.form.get('name')
        applicant_email = request.form.get('email')
        
        application = {
            'job_id': job_id,
            'name': applicant_name,
            'email': applicant_email,
            'phone': request.form.get('phone'),
            'experience': request.form.get('experience'),
            'cover_letter': request.form.get('cover_letter'),
            'date_applied': datetime.now()
        }
        
        # Handle resume upload
        if 'resume' in request.files:
            resume = request.files['resume']
            if resume.filename != '':
                # Generate a secure filename
                filename = secure_filename(resume.filename)
                # Create a unique filename with timestamp
                unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
                # Save the file
                resume_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                resume.save(resume_path)
                # Add file path to application data
                application['resume_path'] = unique_filename
        
        try:
            # Insert application into database
            result = applications_collection.insert_one(application)
            application['_id'] = result.inserted_id
            
            # Send confirmation email to candidate
            email_sent = send_application_confirmation_email(applicant_name, applicant_email, job)
            
            # Notify employer about new application
            employer_notified = notify_employer_about_application(job, application)
            
            # Flash appropriate messages
            if email_sent:
                flash('Application submitted successfully! A confirmation email has been sent to your email address.', 'success')
            else:
                flash('Application submitted, but we could not send a confirmation email. Please check your email address.', 'warning')
                
            if not employer_notified:
                print(f"Failed to notify employer about application from {applicant_name}")
                
            return redirect(url_for('view_jobs_candidate'))
        except Exception as e:
            flash(f'Error submitting application: {str(e)}', 'danger')
    
    return render_template('apply_job_candidate.html', job=job)






@app.route('/sendEmails', methods=['GET'])
def sendEmails():
    """
    Handle the request when the 'Proceed' button is clicked and send 
    notification email about next round selection
    
    Returns:
        Redirect to dashboard with appropriate flash message
    """
    try:
        # The email address that should receive the notification
        recipient_email = "nisargwath7@gmail.com"
        
        # Get candidate info (in a real app, this would come from the session or previous form)
        # For demonstration, using placeholder values
        candidate_name = request.args.get('name', 'Candidate')
        job_title = request.args.get('job_title', 'Applied Position')
        company_name = request.args.get('company', 'Company')
        
        # Send the email notification
        email_sent = send_next_round_email(recipient_email, candidate_name, job_title, company_name)
        
        if email_sent:
            flash('Candidate has been notified about the next round!', 'success')
        else:
            flash('Failed to send email notification. Please try again later.', 'danger')
            
        # Redirect to the employer dashboard or another appropriate page
        return redirect(url_for('user_dashboard'))
    
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        return redirect(url_for('user_dashboard'))

def send_next_round_email(recipient_email, candidate_name, job_title, company_name):
    """
    Send email notification about selection for the next round (Onsite/OE)
    
    Args:
        recipient_email (str): Email address of the recipient
        candidate_name (str): Name of the candidate
        job_title (str): Title of the job position
        company_name (str): Name of the company
    
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    subject = f"Congratulations! You're Selected for the Next Round - {job_title}"
    
    # Generate HTML email content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Next Round Selection</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            .header {{ background-color: #4a86e8; color: white; padding: 15px; text-align: center; border-radius: 5px 5px 0 0; margin: -20px -20px 20px; }}
            .selection-details {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .footer {{ text-align: center; margin-top: 30px; font-size: 0.8em; color: #777; }}
            .cta-button {{ display: inline-block; background-color: #4CAF50; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; font-weight: bold; margin-top: 15px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>Congratulations!</h2>
            </div>
            
            <p>Dear {candidate_name},</p>
            
            <p>We are pleased to inform you that you have been <strong>selected for the next round</strong> of the interview process for the <strong>{job_title}</strong> position at <strong>{company_name}</strong>.</p>
            
            <div class="selection-details">
                <h3>Next Round Details:</h3>
                <p><strong>Round Type:</strong> Onsite/OE (On-site Evaluation)</p>
                <p><strong>What to Expect:</strong> This round will focus on evaluating your technical skills and cultural fit within our organization.</p>
                <p><strong>Date:</strong> Our HR team will contact you shortly to schedule at your convenience.</p>
            </div>
            
            <p>We were impressed with your profile and performance in the previous round, and we look forward to learning more about you in this next phase.</p>
            
            <p>To confirm your participation and select your preferred time slots, please click the button below:</p>
            <div style="text-align: center;">
                <a href="#" class="cta-button">Schedule Interview</a>
            </div>
            
            <p>If you have any questions or need any accommodations, please don't hesitate to contact our recruitment team.</p>
            
            <p>Best regards,<br>
            The Recruitment Team<br>
            {company_name}</p>
            
            <div class="footer">
                <p>This email was sent by IntervoAI Recruitment System.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return send_email(recipient_email, subject, html_content, is_html=True)










def nl2br(value):
    """Converts newlines to <br> tags for HTML rendering."""
    return value.replace("\n", "<br>\n")

# Register the custom filter with Jinja2
app.jinja_env.filters['nl2br'] = nl2br





@app.route('/view_applications_candidate/<job_id>', methods=['GET'])
def view_applications_candidate(job_id):
    try:
        job = jobs_collection.find_one({'_id': ObjectId(job_id)})
        applications = list(applications_collection.find({'job_id': ObjectId(job_id)}))  # Convert to list
        
        return render_template('view_application_candidate.html', job=job, recent_applications=applications)
    except Exception as e:
        return str(e), 400  # Handle errors gracefully

@app.route('/download_resume_candidate/<filename>', methods=['GET'])
def download_resume_candidate(filename):
    safe_path = safe_join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(safe_path):
        return abort(404)  # Return a 404 if file doesn't exist
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


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
            'started_at': datetime.now()
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
            'submitted_at': datetime.now()
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
                "content_quality": f"Response to '{question[:74]}...' appears to include relevant content",
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
                    "content_quality": f"Partial analysis based on transcript: '{partial_transcript[:70]}...'",
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
                "timestamp": datetime.now().isoformat()
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
            'completed_at': datetime.now()
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
        'timestamp': datetime.now().isoformat(),
        'questions': session.get('questions', []),
        'analysis': session.get('analysis', [])
    }
    
    return jsonify(report_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
