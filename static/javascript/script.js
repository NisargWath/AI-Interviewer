// Handle resume upload and auto-fill form fields
if (document.getElementById('resume')) {
    document.getElementById('resume').addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('resume', file);
            
            // Show loading indicator
            const uploadButton = this.parentElement.querySelector('small');
            if (uploadButton) {
                uploadButton.textContent = 'Parsing resume...';
            }
            
            // Send the file to the server for parsing
            fetch('/parse-resume', {
                method: 'POST',
                body: formData,
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Resume parsing failed');
                }
                return response.json();
            })
            .then(data => {
                // Auto-fill the form fields with extracted data
                if (document.getElementById('fullName')) {
                    document.getElementById('fullName').value = data.fullName || '';
                }
                if (document.getElementById('email')) {
                    document.getElementById('email').value = data.email || '';
                }
                if (document.getElementById('phone')) {
                    document.getElementById('phone').value = data.phone || '';
                }
                if (document.getElementById('skills')) {
                    document.getElementById('skills').value = data.skills || '';
                }
                
                // Reset upload message
                if (uploadButton) {
                    uploadButton.textContent = 'Resume parsed successfully!';
                    setTimeout(() => {
                        uploadButton.textContent = 'Upload your resume to auto-fill the form below.';
                    }, 3000);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                // Reset upload message
                if (uploadButton) {
                    uploadButton.textContent = 'Error parsing resume. Please fill in the form manually.';
                }
            });
        }
    });
}

// Handle job application form submission (Registration for job seekers)
if (document.getElementById('jobForm')) {
    document.getElementById('jobForm').addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent the default form submission
        
        // Validate passwords match
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        
        if (password !== confirmPassword) {
            alert('Passwords do not match!');
            return;
        }
        
        const formData = new FormData(this);
        
        // Show loading state
        const submitButton = this.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.textContent;
        submitButton.textContent = 'Submitting...';
        submitButton.disabled = true;
        
        fetch('/register', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                // Show error message
                alert(data.error);
                submitButton.textContent = originalButtonText;
                submitButton.disabled = false;
            } else if (data.redirect) {
                // Show success message before redirect
                alert(data.message || 'Registration successful!');
                // Redirect to the login page after successful registration
                window.location.href = data.redirect;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Registration failed. Please try again.');
            submitButton.textContent = originalButtonText;
            submitButton.disabled = false;
        });
    });
}

// Handle recruiter registration form submission
if (document.getElementById('recruiterForm')) {
    document.getElementById('recruiterForm').addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent the default form submission
        
        // Validate passwords match
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        
        if (password !== confirmPassword) {
            alert('Passwords do not match!');
            return;
        }
        
        // Get form data and convert to JSON
        const formData = new FormData(this);
        const formJson = Object.fromEntries(formData.entries());
        
        // Show loading state
        const submitButton = this.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.textContent;
        submitButton.textContent = 'Submitting...';
        submitButton.disabled = true;
        
        fetch('/recruiter-register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formJson),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                // Show error message
                alert(data.error);
                submitButton.textContent = originalButtonText;
                submitButton.disabled = false;
            } else if (data.redirect) {
                // Show success message before redirect
                alert(data.message || 'Recruiter registration successful!');
                // Redirect to the login page after successful registration
                window.location.href = data.redirect;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Registration failed. Please try again.');
            submitButton.textContent = originalButtonText;
            submitButton.disabled = false;
        });
    });
}

// Handle login form submission (for job seekers)
if (document.getElementById('loginForm')) {
    document.getElementById('loginForm').addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent the default form submission
        
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const userType = document.querySelector('input[name="userType"]').value;
        
        // Show loading state
        const submitButton = this.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.textContent;
        submitButton.textContent = 'Logging in...';
        submitButton.disabled = true;
        
        fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password, userType }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Login failed');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                // Show error message
                alert(data.error);
                submitButton.textContent = originalButtonText;
                submitButton.disabled = false;
            } else if (data.redirect) {
                // Redirect to the appropriate dashboard based on user type
                window.location.href = data.redirect;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Login failed. Please check your credentials and try again.');
            submitButton.textContent = originalButtonText;
            submitButton.disabled = false;
        });
    });
}

// Handle recruiter login form submission
if (document.getElementById('recruiterLoginForm')) {
    document.getElementById('recruiterLoginForm').addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent the default form submission
        
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const userType = 'recruiter'; // Hard-coded for recruiter login form
        
        // Show loading state
        const submitButton = this.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.textContent;
        submitButton.textContent = 'Logging in...';
        submitButton.disabled = true;
        
        fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password, userType }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Login failed');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                // Show error message
                alert(data.error);
                submitButton.textContent = originalButtonText;
                submitButton.disabled = false;
            } else if (data.redirect) {
                // Redirect to the appropriate dashboard
                window.location.href = data.redirect;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Login failed. Please check your credentials and try again.');
            submitButton.textContent = originalButtonText;
            submitButton.disabled = false;
        });
    });
}