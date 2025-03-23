# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import datetime
import markdown
import re 

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")

# Database setup
def init_db():
    conn = sqlite3.connect('learning_pathways.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        learning_style TEXT,
        preferred_study_time INTEGER,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pathways (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        topic TEXT,
        pathway_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pathway_id INTEGER,
        step_id TEXT,
        completed BOOLEAN DEFAULT 0,
        feedback TEXT,
        completed_at TIMESTAMP,
        FOREIGN KEY (pathway_id) REFERENCES pathways (id)
    )
    ''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Helper function to get or create guest user
def get_guest_user():
    if 'user_id' not in session:
        # Create a guest user session
        session['user_id'] = 1  # Using ID 1 for guest user
        session['username'] = 'guest'
    return session['user_id']

@app.route('/')
def index():
    get_guest_user()  # Ensure user session exists
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('learning_pathways.db')
        cursor = conn.cursor()
        
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                        (username, password))
            conn.commit()
            return jsonify({"success": True, "message": "Registration successful!"})
        except sqlite3.IntegrityError:
            return jsonify({"success": False, "message": "Username already exists."})
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('learning_pathways.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and user[1] == password:
            session['user_id'] = user[0]
            session['username'] = username
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "message": "Invalid credentials"})
    
    return render_template('login.html')

@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    user_id = get_guest_user()  # No login check, just get a user ID
    
    if request.method == 'POST':
        learning_style = request.form['learning_style']
        preferred_study_time = request.form['preferred_study_time']
        
        conn = sqlite3.connect('learning_pathways.db')
        cursor = conn.cursor()
        
        # Check if preferences already exist
        cursor.execute("SELECT id FROM preferences WHERE user_id = ?", (user_id,))
        pref = cursor.fetchone()
        
        if pref:
            cursor.execute("""
                UPDATE preferences 
                SET learning_style = ?, preferred_study_time = ?
                WHERE user_id = ?
            """, (learning_style, preferred_study_time, user_id))
        else:
            cursor.execute("""
                INSERT INTO preferences (user_id, learning_style, preferred_study_time)
                VALUES (?, ?, ?)
            """, (user_id, learning_style, preferred_study_time))
        
        conn.commit()
        conn.close()
        
        return jsonify({"success": True, "message": "Preferences saved successfully"})
    
    # Get existing preferences
    conn = sqlite3.connect('learning_pathways.db')
    cursor = conn.cursor()
    cursor.execute("SELECT learning_style, preferred_study_time FROM preferences WHERE user_id = ?", 
                (user_id,))
    prefs = cursor.fetchone()
    conn.close()
    
    return render_template('preferences.html', preferences=prefs)

@app.route('/generate_pathway', methods=['POST'])
def generate_pathway():
    user_id = get_guest_user()  # No login check, just get a user ID
    
    topic = request.form['topic']
    learning_style = request.form['learning_style']
    study_time = request.form['study_time']
    
    # Get user preferences if not provided
    if not learning_style or not study_time:
        conn = sqlite3.connect('learning_pathways.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT learning_style, preferred_study_time 
            FROM preferences 
            WHERE user_id = ?
        """, (user_id,))
        prefs = cursor.fetchone()
        conn.close()
        
        if prefs:
            learning_style = learning_style or prefs[0]
            study_time = study_time or prefs[1]
    
    # Generate learning pathway using Gemini AI
    prompt = f"""
    Create a personalized learning pathway for a student with the following details:
    
    Topic: {topic}
    Learning Style: {learning_style}
    Available Study Time: {study_time} hours per week
    
    Please structure your response as a JSON object with the following format:
    {{
        "topic": "The main topic",
        "overview": "Brief overview of the topic and learning goals",
        "estimated_completion_time": "Total estimated time in weeks",
        "modules": [
            {{
                "id": "module1",
                "title": "Module Title",
                "description": "Brief description of this module",
                "estimated_time": "Time in hours",
                "resources": [
                    {{
                        "type": "video/article/exercise/quiz",
                        "title": "Resource Title",
                        "description": "Brief description",
                        "estimated_time": "Time in minutes"
                    }}
                ],
                "activities": [
                    {{
                        "id": "activity1",
                        "title": "Activity Title",
                        "description": "What to do",
                        "estimated_time": "Time in minutes"
                    }}
                ],
                "assessment": {{
                    "type": "quiz/project/reflection",
                    "description": "Assessment description"
                }}
            }}
        ],
        "next_steps": "Suggestions for what to learn after completing this pathway"
    }}
    
    Adapt the content to the learning style: {learning_style}.
    Make sure all time estimates are realistic for the {study_time} hours per week availability.
    """
    
    try:
        response = model.generate_content(prompt)
        print(response)
        pathway_data = response.text
        
        # Extract JSON from the response
        try:
            # Try to find JSON in the response if it's wrapped in markdown code blocks
            if "```json" in pathway_data:
                json_start = pathway_data.find("```json") + 7
                json_end = pathway_data.find("```", json_start)
                pathway_json = pathway_data[json_start:json_end].strip()
            else:
                pathway_json = pathway_data.strip()
            
            # Validate JSON
            parsed_json = json.loads(pathway_json)
            
            # Save pathway to database
            conn = sqlite3.connect('learning_pathways.db')
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pathways (user_id, topic, pathway_data)
                VALUES (?, ?, ?)
            """, (user_id, topic, json.dumps(parsed_json)))
            
            pathway_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return jsonify({
                "success": True, 
                "pathway_id": pathway_id,
                "pathway": parsed_json
            })
        except json.JSONDecodeError:
            return jsonify({
                "success": False, 
                "message": "Failed to parse AI response as JSON",
                "raw_response": pathway_data
            })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/pathway/<int:pathway_id>')
def view_pathway(pathway_id):
    user_id = get_guest_user()  # No login check, just get a user ID
    
    conn = sqlite3.connect('learning_pathways.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT pathway_data FROM pathways 
        WHERE id = ?
    """, (pathway_id,))  # Removed user_id filter to allow any user to view any pathway
    
    pathway = cursor.fetchone()
    conn.close()
    
    if not pathway:
        return "Pathway not found", 404
    
    pathway_data = json.loads(pathway[0])
    
    return render_template('pathway.html', pathway=pathway_data, pathway_id=pathway_id)

@app.route('/update_progress', methods=['POST'])
def update_progress():
    user_id = get_guest_user()  # No login check, just get a user ID
    
    pathway_id = request.form['pathway_id']
    step_id = request.form['step_id']
    completed = request.form['completed'] == 'true'
    feedback = request.form.get('feedback', '')
    
    conn = sqlite3.connect('learning_pathways.db')
    cursor = conn.cursor()
    
    # Check if progress entry exists
    cursor.execute("""
        SELECT id FROM progress 
        WHERE pathway_id = ? AND step_id = ?
    """, (pathway_id, step_id))
    
    progress = cursor.fetchone()
    
    if progress:
        if completed:
            cursor.execute("""
                UPDATE progress 
                SET completed = ?, feedback = ?, completed_at = ?
                WHERE pathway_id = ? AND step_id = ?
            """, (completed, feedback, datetime.datetime.now(), pathway_id, step_id))
        else:
            cursor.execute("""
                UPDATE progress 
                SET completed = ?, feedback = ?, completed_at = NULL
                WHERE pathway_id = ? AND step_id = ?
            """, (completed, feedback, pathway_id, step_id))
    else:
        if completed:
            cursor.execute("""
                INSERT INTO progress (pathway_id, step_id, completed, feedback, completed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (pathway_id, step_id, completed, feedback, datetime.datetime.now()))
        else:
            cursor.execute("""
                INSERT INTO progress (pathway_id, step_id, completed, feedback)
                VALUES (?, ?, ?, ?)
            """, (pathway_id, step_id, completed, feedback))
    
    conn.commit()
    conn.close()
    
    return jsonify({"success": True})

@app.route('/get_progress/<int:pathway_id>')
def get_progress(pathway_id):
    user_id = get_guest_user()  # No login check, just get a user ID
    
    conn = sqlite3.connect('learning_pathways.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT step_id, completed, feedback, completed_at 
        FROM progress 
        WHERE pathway_id = ?
    """, (pathway_id,))
    
    progress_data = cursor.fetchall()
    conn.close()
    
    progress = {}
    for item in progress_data:
        progress[item[0]] = {
            "completed": bool(item[1]),
            "feedback": item[2],
            "completed_at": item[3]
        }
    
    return jsonify({"success": True, "progress": progress})

@app.route('/dashboard')
def dashboard():
    user_id = get_guest_user()  # No login check, just get a user ID
    
    conn = sqlite3.connect('learning_pathways.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, topic, created_at 
        FROM pathways 
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (user_id,))
    
    pathways = cursor.fetchall()
    
    # Get progress for each pathway
    pathways_with_progress = []
    for pathway in pathways:
        pathway_id = pathway[0]
        
        # Get total steps
        cursor.execute("""
            SELECT pathway_data FROM pathways WHERE id = ?
        """, (pathway_id,))
        pathway_data = json.loads(cursor.fetchone()[0])
        
        total_steps = 0
        for module in pathway_data.get('modules', []):
            # Count resources
            total_steps += len(module.get('resources', []))
            # Count activities
            total_steps += len(module.get('activities', []))
            # Count assessment (if exists)
            if 'assessment' in module:
                total_steps += 1
        
        # Get completed steps
        cursor.execute("""
            SELECT COUNT(*) FROM progress 
            WHERE pathway_id = ? AND completed = 1
        """, (pathway_id,))
        completed_steps = cursor.fetchone()[0]
        
        # Calculate progress percentage
        progress_percentage = 0
        if total_steps > 0:
            progress_percentage = (completed_steps / total_steps) * 100
        
        pathways_with_progress.append({
            'id': pathway_id,
            'topic': pathway[1],
            'created_at': pathway[2],
            'total_steps': total_steps,
            'completed_steps': completed_steps,
            'progress_percentage': progress_percentage
        })
    
    conn.close()
    
    return render_template('dashboard.html', pathways=pathways_with_progress)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Modify the chatbot route to include pathway context
@app.route('/chatbot/<int:pathway_id>')
def chatbot(pathway_id=None):
    # Get pathway data if pathway_id is provided
    pathway_data = None
    if pathway_id:
        conn = sqlite3.connect('learning_pathways.db')
        cursor = conn.cursor()
        cursor.execute("SELECT pathway_data FROM pathways WHERE id = ?", (pathway_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            pathway_data = json.loads(result[0])
    
    return render_template('chatbot.html', pathway_id=pathway_id, pathway_data=pathway_data)

# Update the chat endpoint to include pathway context in the prompt
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    pathway_id = request.json.get('pathway_id')
    
    pathway_context = ""
    if pathway_id:
        # Fetch pathway data from the database
        conn = sqlite3.connect('learning_pathways.db')
        cursor = conn.cursor()
        cursor.execute("SELECT pathway_data FROM pathways WHERE id = ?", (pathway_id,))
        result = cursor.fetchone()
        print(result)
        conn.close()
        
        if result:
            pathway_data = json.loads(result[0])
            # Create a summary of the pathway to use as context
            print(pathway_data.get('topic'))
            pathway_context = f"""
            generate text without bolding the text  in 100-200 words
            Context for this conversation:
            The user is working on a learning pathway with the following details:
            Topic: {pathway_data.get('topic')}
            Overview: {pathway_data.get('overview')}
            Modules: {', '.join([module.get('title') for module in pathway_data.get('modules', [])])}
            """
    
    # Create a chat session
    chat_session = model.start_chat(history=[])
    
    # Generate a response with pathway context
    full_prompt = f"{pathway_context}\n\nUser question: {user_message}\n\nPlease provide a helpful response based on the learning pathway context. Keep your answer concise and focused on the user's learning journey."
    
    response = chat_session.send_message(full_prompt)
    
    return jsonify({'response': response.text})

if __name__ == '__main__':
    app.run(debug=True, port=8000) 