// Add CSRF token handling at the top of your script
const csrfToken = document.querySelector('meta[name="csrf-token"]').content;

// Your existing FAQ code with AJAX modifications
document.querySelectorAll('.faq-item h3').forEach(question => {
    question.addEventListener('click', function() {
        const answer = this.nextElementSibling;
        const toggle = this.querySelector('.toggle');

        // Close all other answers
        document.querySelectorAll('.faq-answer').forEach(item => {
            if (item !== answer) {
                item.style.maxHeight = null;
                item.previousElementSibling.querySelector('.toggle').textContent = '+';
            }
        });

        // Toggle current answer
        if (answer.style.maxHeight) {
            answer.style.maxHeight = null;
            toggle.textContent = '+';
        } else {
            answer.style.maxHeight = answer.scrollHeight + "px";
            toggle.textContent = '−';
        }

        // If you need to send this interaction to the server
        fetch('/log-interaction', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify({
                question: this.innerText,
                action: answer.style.maxHeight ? 'opened' : 'closed'
            })
        });
    });
});

// Add this for jQuery AJAX compatibility (if you're using jQuery)
if (typeof jQuery !== 'undefined') {
    $.ajaxSetup({
        headers: {
            'X-CSRFToken': csrfToken
        }
    });
}
