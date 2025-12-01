// Floating particle background
function createParticles() {
    const particlesContainer = document.getElementById('particles');
    if (!particlesContainer) return;

    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.width = (Math.random() * 4 + 2) + 'px';
        particle.style.height = particle.style.width;
        particle.style.animationDelay = Math.random() * 15 + 's';
        particlesContainer.appendChild(particle);
    }
}

document.addEventListener("DOMContentLoaded", function () {
    createParticles();

    const form = document.getElementById("carForm");
    if (form) {
        // --- ENHANCED CLIENT-SIDE VALIDATION ---
        form.addEventListener("submit", function (e) {
            let valid = true;
            const inputs = form.querySelectorAll("input, select");

            inputs.forEach(input => {
                // Check native validity
                if (!input.checkValidity()) {
                    valid = false;
                    input.classList.add('error');
                } else {
                    input.classList.remove('error');
                }
            });

            if (!valid) {
                // Prevent form submission if validation fails
                e.preventDefault();
                // Optionally show an alert (browser's default required message will also show)
                // alert("⚠ Please fill all required fields before submitting!"); 
            }
            // If valid, the form proceeds normally
        });

        // Add 'error' class on blur if invalid to show immediate feedback
        const inputs = form.querySelectorAll("input, select");
        inputs.forEach(input => {
            input.addEventListener('blur', () => {
                if (!input.checkValidity()) {
                    input.classList.add('error');
                } else {
                    input.classList.remove('error');
                }
            });
        });
    }

    // Counter animation for result page
    const priceElement = document.getElementById('priceValue');
    const perfElement = document.getElementById('perfValue');
    const progressBar = document.getElementById('progressBar');
    const perfPercentageElement = document.getElementById('perfPercentage');


    if (priceElement && perfElement) {
        // Remove non-numeric characters for animation calculation
        // The price value now contains the '$' symbol
        const priceValue = parseFloat(priceElement.textContent.replace(/[^0-9.]/g, "")); 
        const perfValue = parseFloat(perfElement.textContent.replace(/[^0-9.]/g, "")); 
        
        // Clear initial text to show animation from start
        priceElement.textContent = '$0'; // UPDATED: Changed prefix to $
        perfElement.textContent = '0';
        
        // Animate price
        // PREFIX is now '$'
        animateValue(priceElement, 0, priceValue, 1500, '$', true, function() { // UPDATED: Prefix is '$'
            // Animate performance value after price is done
            animateValue(perfElement, 0, perfValue, 1500, '', false, function() {
                // Animate progress bar and percentage after performance value is done
                if (progressBar) {
                    progressBar.style.width = perfValue + '%';
                }
                if (perfPercentageElement) {
                    // Update percentage with the final value
                    perfPercentageElement.textContent = perfValue.toFixed(0) + '%'; 
                }
                // Add the /100 visual text after the animation is complete
                perfElement.textContent = perfValue.toFixed(0) + '/100'; 
            });
        });
    }
});

function animateValue(element, start, end, duration, prefix = '', isPrice = false, callback) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        
        let value = (progress * (end - start) + start);
        
        if (isPrice) {
            // UPDATED: Use US locale formatting (with commas)
            value = value.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 }); 
        } else {
            value = value.toFixed(0); // Performance score is integer
        }

        element.textContent = prefix + value;
        
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
            // Ensure final value is accurate and correctly formatted
            if (isPrice) {
                 // UPDATED: Use US locale formatting (with commas)
                 element.textContent = prefix + end.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
            } else {
                 element.textContent = end.toFixed(0);
            }

            if (callback) {
                callback();
            }
        }
    };
    window.requestAnimationFrame(step);
}