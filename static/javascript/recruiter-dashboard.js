document.addEventListener('DOMContentLoaded', function() {
    // Initialize Chart.js charts
    const applicationsCtx = document.getElementById('applicationsChart').getContext('2d');
    const applicationsChart = new Chart(applicationsCtx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
            datasets: [{
                label: 'Applications',
                data: [65, 59, 80, 81, 56, 55, 40],
                fill: false,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    const sourcesCtx = document.getElementById('sourcesChart').getContext('2d');
    const sourcesChart = new Chart(sourcesCtx, {
        type: 'doughnut',
        data: {
            labels: ['LinkedIn', 'Indeed', 'Company Website', 'Referrals'],
            datasets: [{
                label: 'Candidate Sources',
                data: [300, 150, 100, 50],
                backgroundColor: [
                    'rgb(255, 99, 132)',
                    'rgb(54, 162, 235)',
                    'rgb(255, 205, 86)',
                    'rgb(75, 192, 192)'
                ],
                hoverOffset: 4
            }]
        }
    });

    // Tab functionality
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            button.classList.add('active');
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });

    // Sidebar navigation
    const sidebarLinks = document.querySelectorAll('.sidebar-nav a');
    const contentSections = document.querySelectorAll('.content-section');

    sidebarLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tab = link.parentElement.getAttribute('data-tab');
            sidebarLinks.forEach(l => l.parentElement.classList.remove('active'));
            contentSections.forEach(section => section.classList.remove('active'));
            link.parentElement.classList.add('active');
            document.getElementById(`${tab}-section`).classList.add('active');
        });
    });
});