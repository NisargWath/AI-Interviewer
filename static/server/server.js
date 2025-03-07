const express = require('express');
const mongoose = require('mongoose');
const multer = require('multer');
const path = require('path');
const pdfParse = require('pdf-parse');
const fs = require('fs');
const bcrypt = require('bcrypt');
const session = require('express-session');

const app = express();
const port = 3000;

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/jobapp', {
    useNewUrlParser: true,
    useUnifiedTopology: true,
});

// Define schema for job seekers
const userSchema = new mongoose.Schema({
    fullName: String,
    email: { type: String, unique: true },
    password: String,
    phone: String,
    experience: Number,
    skills: String,
    resume: String, // Store the path to the resume file
    userType: { type: String, default: 'jobseeker' }
});

// Define schema for job recruiters
const recruiterSchema = new mongoose.Schema({
    companyName: String,
    fullName: String,
    email: { type: String, unique: true },
    password: String,
    phone: String,
    industry: String,
    companySize: String,
    companyLocation: String,
    userType: { type: String, default: 'recruiter' }
});

// Define schema for job postings
const jobSchema = new mongoose.Schema({
    title: String,
    company: String,
    location: String,
    description: String,
    requirements: String,
    salary: String,
    recruiter: { type: mongoose.Schema.Types.ObjectId, ref: 'JobRecruiter' },
    postedDate: { type: Date, default: Date.now },
});

const User = mongoose.model('jobseekers', userSchema);
const Recruiter = mongoose.model('jobrecruiters', recruiterSchema);
const Job = mongoose.model('jobs', jobSchema);

// Middleware for session management
app.use(session({
    secret: 'your-secret-key',
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false } // Set to true if using HTTPS
}));

// Middleware to serve static files
app.use(express.static('public'));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir);
}

// Set up multer for file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/');
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + path.extname(file.originalname));
    },
});

const upload = multer({ storage: storage });

// ----- JOB SEEKER ROUTES -----

// Route to serve the registration page (index)
app.get('/', (req, res) => {
    res.sendFile(__dirname + '/public/index.html');
});

// Route to serve the login page
app.get('/login', (req, res) => {
    res.sendFile(__dirname + '/public/login.html');
});

// Route to handle login form submission (for both jobseekers and recruiters)
app.post('/login', async (req, res) => {
    const { email, password, userType } = req.body;
    
    try {
        let user;
        
        // Check if login is for jobseeker or recruiter
        if (userType === 'recruiter') {
            user = await Recruiter.findOne({ email });
        } else {
            user = await User.findOne({ email });
        }

        if (!user) {
            return res.status(400).json({ error: 'Invalid email or password' });
        }

        const isPasswordValid = await bcrypt.compare(password, user.password);

        if (!isPasswordValid) {
            return res.status(400).json({ error: 'Invalid email or password' });
        }

        // Set session data
        req.session.user = {
            id: user._id,
            email: user.email,
            name: user.fullName || user.companyName,
            userType: user.userType
        };
        
        console.log('User logged in:', req.session.user);
        
        // Redirect based on user type
        if (user.userType === 'recruiter') {
            return res.json({ 
                message: 'Login successful', 
                redirect: '/recruiter-dashboard',
                userType: 'recruiter'
            });
        } else {
            return res.json({ 
                message: 'Login successful', 
                redirect: '/home',
                userType: 'jobseeker'
            });
        }
    } catch (error) {
        console.error('Error during login:', error);
        return res.status(500).json({ error: 'Error during login' });
    }
});

// Route to serve the home page (jobseeker dashboard)
app.get('/home', (req, res) => {
    if (!req.session.user) {
        return res.redirect('/login');
    }
    if (req.session.user.userType === 'recruiter') {
        return res.redirect('/recruiter-dashboard');
    }
    res.sendFile(__dirname + '/public/home.html');
});

// Route to handle logout
app.get('/logout', (req, res) => {
    req.session.destroy(err => {
        if (err) {
            console.error('Error during logout:', err);
            return res.status(500).json({ error: 'Error during logout' });
        }
        res.redirect('/login');
    });
});

// Route to handle registration form submission (jobseeker)
app.post('/register', upload.single('resume'), async (req, res) => {
    const { fullName, email, password, confirmPassword, phone, experience, skills } = req.body;

    if (password !== confirmPassword) {
        return res.status(400).json({ error: 'Passwords do not match' });
    }

    try {
        const hashedPassword = await bcrypt.hash(password, 10);

        const newUser = new User({
            fullName,
            email,
            password: hashedPassword,
            phone,
            experience,
            skills,
            resume: req.file ? req.file.path : null,
            userType: 'jobseeker'
        });

        await newUser.save();
        res.json({ message: 'Registration successful', redirect: '/login' });
    } catch (error) {
        console.error('Error during registration:', error);
        res.status(500).json({ error: 'Error during registration' });
    }
});

// Route to handle resume parsing
app.post('/parse-resume', upload.single('resume'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }
    
    const resumePath = req.file.path;

    try {
        const dataBuffer = fs.readFileSync(resumePath);
        const data = await pdfParse(dataBuffer);

        // Extract details from the resume text
        const extractedData = {
            fullName: extractName(data.text),
            email: extractEmail(data.text),
            phone: extractPhone(data.text),
            skills: extractSkills(data.text),
        };

        res.json(extractedData);
    } catch (error) {
        console.error('Error parsing resume:', error);
        res.status(500).json({ error: 'Error parsing resume' });
    }
});

// ----- RECRUITER ROUTES -----

// Route to serve the recruiter registration page
app.get('/recruiter-register', (req, res) => {
    res.sendFile(__dirname + '/public/recruiter-register.html');
});

// Route to serve the recruiter login page
app.get('/recruiter-login', (req, res) => {
    res.sendFile(__dirname + '/public/recruiter-login.html');
});

// Route to serve the recruiter dashboard
app.get('/recruiter-dashboard', (req, res) => {
    if (!req.session.user) {
        return res.redirect('/recruiter-login');
    }
    if (req.session.user.userType !== 'recruiter') {
        return res.redirect('/login');
    }
    res.sendFile(__dirname + '/public/recruiter-dashboard.html');
});

// Route to handle recruiter registration
app.post('/recruiter-register', async (req, res) => {
    const { 
        companyName, 
        fullName, 
        email, 
        password, 
        confirmPassword, 
        phone, 
        industry, 
        companySize,
        companyLocation 
    } = req.body;

    if (password !== confirmPassword) {
        return res.status(400).json({ error: 'Passwords do not match' });
    }

    try {
        const hashedPassword = await bcrypt.hash(password, 10);

        const newRecruiter = new Recruiter({
            companyName,
            fullName,
            email,
            password: hashedPassword,
            phone,
            industry,
            companySize,
            companyLocation,
            userType: 'recruiter'
        });

        await newRecruiter.save();
        res.json({ message: 'Recruiter registration successful', redirect: '/recruiter-login' });
    } catch (error) {
        console.error('Error during recruiter registration:', error);
        res.status(500).json({ error: 'Error during registration' });
    }
});

// Route to handle job posting
app.post('/post-job', async (req, res) => {
    if (!req.session.user || req.session.user.userType !== 'recruiter') {
        return res.status(401).json({ error: 'Unauthorized' });
    }

    const { title, company, location, description, requirements, salary } = req.body;

    try {
        const newJob = new Job({
            title,
            company,
            location,
            description,
            requirements,
            salary,
            recruiter: req.session.user.id
        });

        await newJob.save();
        res.json({ message: 'Job posted successfully', jobId: newJob._id });
    } catch (error) {
        console.error('Error during job posting:', error);
        res.status(500).json({ error: 'Error during job posting' });
    }
});

// Route to get all jobs posted by a recruiter
app.get('/recruiter-jobs', async (req, res) => {
    if (!req.session.user || req.session.user.userType !== 'recruiter') {
        return res.status(401).json({ error: 'Unauthorized' });
    }

    try {
        const jobs = await Job.find({ recruiter: req.session.user.id }).sort({ postedDate: -1 });
        res.json(jobs);
    } catch (error) {
        console.error('Error fetching recruiter jobs:', error);
        res.status(500).json({ error: 'Error fetching jobs' });
    }
});

// Route to get all jobs (for job seekers)
app.get('/jobs', async (req, res) => {
    try {
        const jobs = await Job.find().sort({ postedDate: -1 });
        res.json(jobs);
    } catch (error) {
        console.error('Error fetching jobs:', error);
        res.status(500).json({ error: 'Error fetching jobs' });
    }
});

// Helper functions to extract data from resume text
function extractName(text) {
    // Basic example: Assume the first line is the name
    return text.split('\n')[0].trim();
}

function extractEmail(text) {
    const emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/;
    const match = text.match(emailRegex);
    return match ? match[0] : null;
}

function extractPhone(text) {
    const phoneRegex = /(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/;
    const match = text.match(phoneRegex);
    return match ? match[0] : null;
}

function extractSkills(text) {
    // Basic example: Look for common skills
    const skills = ['JavaScript', 'Node.js', 'MongoDB', 'React', 'Python'];
    const foundSkills = skills.filter(skill => text.includes(skill));
    return foundSkills.join(', ');
}

// API to get current user info
app.get('/api/user', (req, res) => {
    if (!req.session.user) {
        return res.status(401).json({ error: 'Not logged in' });
    }
    res.json({
        id: req.session.user.id,
        name: req.session.user.name,
        email: req.session.user.email,
        userType: req.session.user.userType
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});