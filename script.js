document.addEventListener('DOMContentLoaded', () => {
    const popupContainer = document.getElementById('popup-container');
    const submitCountBtn = document.getElementById('submit-count');
    const drugCountInput = document.getElementById('drug-count-input');
    const drugNameSection = document.getElementById('drug-name-section');
    const drugNameForm = document.getElementById('drug-name-form');
    const checkSafetyBtn = document.getElementById('check-safety-btn');
    const resultSection = document.getElementById('result-section');
    const safetyMessageDiv = document.getElementById('safety-message');
    const startOverBtn = document.getElementById('start-over-btn');

    let drugCount = 0;

    // --- STEP 1: Handle initial count submission ---
    submitCountBtn.addEventListener('click', () => {
        drugCount = parseInt(drugCountInput.value);

        if (isNaN(drugCount) || drugCount < 0) {
            alert("Please enter a valid number (0 or greater).");
            return;
        }

        // 1. Animate the popup out (CSS transition/class can be added for smoother effect)
        popupContainer.classList.add('hidden'); 

        // 2. Generate the drug name input fields
        generateDrugInputs(drugCount);

        // 3. Show the next section
        drugNameSection.classList.remove('hidden');
    });

    /**
     * Generates input fields for drug names based on the count.
     * @param {number} count The number of drugs to generate inputs for.
     */
    function generateDrugInputs(count) {
        drugNameForm.innerHTML = ''; // Clear previous inputs
        if (count === 0) {
            // Special case for 0 drugs
            drugNameForm.innerHTML = '<p style="font-style: italic;">No drugs to list. Proceed to safety check.</p>';
            return;
        }

        for (let i = 1; i <= count; i++) {
            const label = document.createElement('label');
            label.setAttribute('for', `drug-name-${i}`);
            label.textContent = `Name of Drug #${i}:`;

            const input = document.createElement('input');
            input.setAttribute('type', 'text');
            input.setAttribute('id', `drug-name-${i}`);
            input.setAttribute('class', 'drug-name-input');
            input.setAttribute('placeholder', `e.g., Aspirin, Tylenol`);
            input.required = true;

            drugNameForm.appendChild(label);
            drugNameForm.appendChild(input);
        }
    }


    // --- STEP 2: Handle the "Simulate Check" button ---
    checkSafetyBtn.addEventListener('click', (e) => {
        e.preventDefault();

        // Simple validation check (ensure all required fields are filled if count > 0)
        if (drugCount > 0) {
            const inputs = drugNameForm.querySelectorAll('.drug-name-input');
            let allFilled = true;
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    allFilled = false;
                }
            });

            if (!allFilled) {
                alert("Please enter the name for all listed drugs.");
                return;
            }
        }

        // Hide the input section
        drugNameSection.classList.add('hidden');

        // Display the crucial **DISCLAIMER**
        const disclaimer = `
            🛑 **IMPORTANT SAFETY NOTICE!** 🛑

            <p>Your health and safety are paramount. **DO NOT rely on this simulation for medical decisions.**</p>
        `;
        safetyMessageDiv.innerHTML = disclaimer;
        
        // Show the result section
        resultSection.classList.remove('hidden');
    });


    // --- STEP 3: Handle the "Start Over" button ---
    startOverBtn.addEventListener('click', () => {
        // Reset state
        drugCountInput.value = 0;
        resultSection.classList.add('hidden');
        
        // Show the initial popup again
        popupContainer.classList.remove('hidden');
        drugNameSection.classList.add('hidden'); // Ensure this is hidden for a clean restart
    });

});
