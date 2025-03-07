document.addEventListener("DOMContentLoaded", function () {
  // Sample questions array - in a real app, this would come from your database
  const questions = [
    "I find it challenging to draw astute conclusions from muddled information.",
    "I prefer structured environments with clear guidelines.",
    "Taking calculated risks energizes me.",
    "I find it easy to empathize with others' perspectives.",
    "I'm comfortable making decisions with incomplete information.",
  ];

  let currentQuestion = 0;
  const answers = [];

  const questionElement = document.getElementById("question");
  const options = document.querySelectorAll(".option");
  const backButton = document.getElementById("backButton");
  const nextButton = document.getElementById("nextButton");
  const progressFill = document.getElementById("progressFill");

  // Update progress bar
  function updateProgress() {
    const progress = (currentQuestion / questions.length) * 100;
    progressFill.style.width = `${progress}%`;
  }

  // Load question
  function loadQuestion() {
    questionElement.textContent = questions[currentQuestion];

    // Deselect all options
    options.forEach((option) => {
      option.classList.remove("selected");

      // If there's a saved answer for this question, select it
      if (
        answers[currentQuestion] &&
        option.value == answers[currentQuestion]
      ) {
        option.classList.add("selected");
      }
    });

    // Update back button state
    backButton.style.visibility = currentQuestion === 0 ? "hidden" : "visible";

    // Update progress
    updateProgress();

    // Add animation
    questionElement.style.opacity = "0";
    setTimeout(() => {
      questionElement.style.opacity = "1";
    }, 50);
  }

  // Initialize
  loadQuestion();

  // Option click handler
  options.forEach((option) => {
    option.addEventListener("click", function () {
      // Deselect all options
      options.forEach((opt) => opt.classList.remove("selected"));

      // Select clicked option
      this.classList.add("selected");

      // Save answer
      answers[currentQuestion] = this.value;

      // Auto-advance after a short delay
      setTimeout(() => {
        if (currentQuestion < questions.length - 1) {
          currentQuestion++;
          loadQuestion();
        } else {
          completeTest();
        }
      }, 500);
    });
  });

  // Back button handler
  backButton.addEventListener("click", function () {
    if (currentQuestion > 0) {
      currentQuestion--;
      loadQuestion();
    }
  });

  // Next button handler
  nextButton.addEventListener("click", function () {
    if (answers[currentQuestion]) {
      // If question is answered, go to next
      if (currentQuestion < questions.length - 1) {
        currentQuestion++;
        loadQuestion();
      } else {
        completeTest();
      }
    } else {
      // Highlight options to indicate selection needed
      document.querySelector(".options").classList.add("highlight");
      setTimeout(() => {
        document.querySelector(".options").classList.remove("highlight");
      }, 500);
    }
  });

  // Test completion
  function completeTest() {
    // In a real app, you would send the answers to your server here

    // Clear the card content
    document.querySelector(".card-header").innerHTML = "";
    document.querySelector(".card-content").innerHTML = "";

    // Show completion message
    const completionMessage = document.createElement("div");
    completionMessage.className = "completion-message";
    completionMessage.innerHTML = `
            <i class="fas fa-check-circle" style="font-size: 3rem; color: var(--primary); margin-bottom: 1rem;"></i>
            <h2>Assessment Complete!</h2>
            <p>Thank you for completing the Behavioral Analysis Test.</p>
            <p>Your results are being processed.</p>
              <a href="/phase1" class="signup-btn" id="signup-btn">
              <span>Next Round</span>
              →
            </a>
        `;

    document.querySelector(".card-header").appendChild(completionMessage);

    // Update progress to 100%
    progressFill.style.width = "100%";
  }
});
