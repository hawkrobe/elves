// Main Experiment Script

// Initialize jsPsych
const jsPsych = initJsPsych({
  on_finish: function() {
    // Save data when experiment finishes
    jsPsych.data.displayData('csv');
  }
});

// Get participant ID
const participantID = getParticipantID();

// Add participant ID to all data
jsPsych.data.addProperties({
  participant_id: participantID,
  experiment_name: EXPERIMENT_CONFIG.EXPERIMENT_NAME,
  experiment_version: EXPERIMENT_CONFIG.EXPERIMENT_VERSION
});

// Note: createCompassHTML() function is defined in compass.js

// ============================================
// Exposure Trial Data
// ============================================
// Array of 8 exposure trials (one for each direction)
// TODO: This will be generated dynamically from trial generation script
// For now, using placeholder data with evenly spaced angles
const exposure_trials_data = [
  { angle: 15, label: 'pim' },    // 0 degrees + 15 offset
  { angle: 60, label: 'dak' },    // 45 degrees + 15 offset
  { angle: 105, label: 'gled' },  // 90 degrees + 15 offset
  { angle: 150, label: 'yeen' },  // 135 degrees + 15 offset
  { angle: 195, label: 'peka' },  // 180 degrees + 15 offset
  { angle: 240, label: 'sarp' },  // 225 degrees + 15 offset
  { angle: 285, label: 'grah' },  // 270 degrees + 15 offset
  { angle: 330, label: 'noobda' } // 315 degrees + 15 offset
];

const training_trials_data = [
    { angle: 45, leftLabel: 'pim', rightLabel: 'dak', target: 'pim' },
    { angle: 135, leftLabel: 'yeen', rightLabel: 'peka', target: 'yeen' },
    { angle: 225, leftLabel: 'sarp', rightLabel: 'grah', target: 'sarp' },
    { angle: 315, leftLabel: 'noobda', rightLabel: 'clate', target: 'noobda' }
  ];

// ============================================
// Phase 2: Consent Page
// ============================================
const consent_trial = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `
    <div class="consent-text">
      <h2>Informed Consent</h2>
      <p><strong>PLACEHOLDER CONSENT TEXT</strong></p>
      <p>This is placeholder consent text. The actual consent text will be provided by the lab and will be inserted here.</p>
      <p>Please read the consent information carefully. By clicking "I agree", you indicate that you understand the information and agree to participate in this research study.</p>
      <p>If you do not agree, please click "I do not agree" and the experiment will end.</p>
    </div>
  `,
  choices: ['I agree', 'I do not agree'],
  button_html: function(choice, choice_index) {
    const buttonClass = choice_index === 0 ? 'consent-button agree' : 'consent-button disagree';
    return `<button class="${buttonClass}">${choice}</button>`;
  },
  data: {
    trial_type: 'consent'
  },
  on_finish: function(data) {
    // Record consent response
    // response is the index of the button clicked (0 = "I agree", 1 = "I do not agree")
    data.consent_response = data.response === 0 ? 'agree' : 'disagree';
    data.consent_timestamp = new Date().toISOString();
    
    // If participant does not agree, end experiment
    if (data.response === 1) { // "I do not agree" is the second button (index 1)
      jsPsych.endExperiment(`
        <div class="instruction-text">
          <h2>Thank you</h2>
          <p>You have chosen not to participate. Thank you for your time.</p>
        </div>
      `);
    }
  }
};

// ============================================
// Phase 3: Introduction Page
// ============================================
const introduction_trial = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: `
    <div class="instruction-text">
      <h2>Welcome to the Elves Treasure Hunt Experiment</h2>
      <p>Thank you for your participation. In order to complete today's task, you're going to play a game. You've been asked to be a guide for a squad of Elves looking to find buried treasure. Their compass navigator has gone missing and there's not much time before winter makes the hunt impossible. In exchange for your navigating, they'll give you a cut of their prize.</p>
      <p>Here's the situation. The worker elves only know Elvish. It's your job to learn to read the compass as quickly and accurately as possible, and use what you know to guide the Elves to treasure. First you'll learn how to give directions. Just as soon as you pass basic navigating it's off to hunt for treasure. Good luck!</p>
      <p class="prompt-text">Press any key to continue</p>
    </div>
  `,
  choices: 'ALL_KEYS',
  data: {
    trial_type: 'introduction'
  }
};

// ============================================
// Phase 4a: Initial Exposure/Naming Instruction Page
// ============================================
const exposure_instructions_trial = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: `
    <div class="instruction-text">
      <h2>Learning Elvish Directions</h2>
      <p>The Elves use 8 principal directions to navigate. You will use those directions to help them find the treasure. You're about to see the names for each direction, and practice giving directions by typing in the name of the direction.</p>
      <p class="prompt-text">Press any key to begin</p>
    </div>
  `,
  choices: 'ALL_KEYS',
  data: {
    trial_type: 'exposure_instructions'
  }
};

// ============================================
// Exposure Trial Template
// ============================================
// Template for exposure trials - will be used with timeline variables
// Each trial repeats until participant gets it correct (matching PsychoPy behavior)
const exposure_trial_template = {
  timeline: [
    {
      type: jsPsychSurveyHtmlForm,
      html: function() {
        // Get trial data from timeline variables
        // Use evaluateTimelineVariable for function contexts
        const angle = jsPsych.evaluateTimelineVariable('angle');
        const label = jsPsych.evaluateTimelineVariable('label');
        
        // Create compass HTML using the helper function
        const compassHTML = createCompassHTML(angle);
        
        // Return HTML with compass and form
        return `
          <div class="compass-container">
            ${compassHTML}
            <p class="compass-instruction">This direction is called <span class="compass-label">${label}</span></p>
            <p style="font-size: 16px; margin: 20px 0;">Type the name of this direction:</p>
            <input type="text" id="exposure-response" name="exposure-response" class="compass-input" autocomplete="off" autofocus />
          </div>
        `;
      },
      autofocus: 'exposure-response',
      data: {
        trial_type: 'exposure'
      },
      on_finish: function(data) {
        // Get target from timeline variable
        // Use evaluateTimelineVariable for function contexts
        const target = jsPsych.evaluateTimelineVariable('label');
        const angle = jsPsych.evaluateTimelineVariable('angle');
        
        // Check if response is correct
        const response = data.response['exposure-response'].toLowerCase().trim();
        data.is_correct = response === target.toLowerCase() ? 1 : 0;
        data.response_text = response;
        data.target = target;
        data.angle = angle;
      }
    }
  ],
  loop_function: function(data) {
    // Repeat trial if participant got it wrong
    // data.values() returns an array of all trial data in this timeline
    const values = data.values();
    if (values.length === 0) return false;
    const lastTrial = values[values.length - 1];
    return lastTrial.is_correct === 0; // Repeat if incorrect
  }
};

// ============================================
// Exposure Trials Timeline
// ============================================
// Create timeline with all 8 exposure trials
// Each trial will loop until correct (handled by loop_function above)
const exposure_trials = {
  timeline: [exposure_trial_template],
  timeline_variables: exposure_trials_data,
  randomize_order: false // Show in order, matching PsychoPy
};

// ============================================
// Phase 4b: Learning Trials Instruction Page
// ============================================
const learning_instructions_trial = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: `
    <div class="instruction-text">
      <h2>Checking Your Progress</h2>
      <p>Let's check in to see how much Elvish you've learned. You will see a direction and two potential words for that direction. Type the word that you think is correct. We'll go back to practicing Elvish if you haven't learned all the words yet.</p>
      <p class="prompt-text">Press any key to begin the training trials</p>
    </div>
  `,
  choices: 'ALL_KEYS',
  data: {
    trial_type: 'learning_instructions'
  }
};

// ============================================
// Training Trial Template
// ============================================
// Template for training trials - will be used with timeline variables
// Each trial repeats until participant gets it correct (matching PsychoPy behavior)
const training_trial_template = {
    timeline: [
      {
        type: jsPsychSurveyHtmlForm,
        html: function() {
          // Get trial data from timeline variables
          // Use evaluateTimelineVariable for function contexts
          const angle = jsPsych.evaluateTimelineVariable('angle');
          const leftLabel = jsPsych.evaluateTimelineVariable('leftLabel');
          const rightLabel = jsPsych.evaluateTimelineVariable('rightLabel');
          const target = jsPsych.evaluateTimelineVariable('target');
          
          // Create compass HTML using the helper function
          const compassHTML = createCompassHTML(angle);
          
          // Return HTML with compass and form
          return `
            <div class="compass-container">
              ${compassHTML}
              <p class="compass-instruction">Which label? <span class="compass-label">${leftLabel}</span> or <span class="compass-label">${rightLabel}</span></p>
              <p style="font-size: 16px; margin: 20px 0;">Type the name of the correct label:</p>
              <input type="text" id="training-response" name="training-response" class="compass-input" autocomplete="off" autofocus />
            </div>
          `;
        },
        autofocus: 'training-response',
        data: {
          trial_type: 'training'
        },
        on_finish: function(data) {
          // Get target from timeline variable
          // Use evaluateTimelineVariable for function contexts
          const target = jsPsych.evaluateTimelineVariable('target');
          const angle = jsPsych.evaluateTimelineVariable('angle');
          
          // Check if response is correct
          const response = data.response['training-response'].toLowerCase().trim();
          data.is_correct = response === target.toLowerCase() ? 1 : 0;
          data.response_text = response;
          data.target = target;
          data.angle = angle;
        }
      }
    ],
    loop_function: function(data) {
      // Repeat trial if participant got it wrong
      // data.values() returns an array of all trial data in this timeline
      const values = data.values();
      if (values.length === 0) return false;
      const lastTrial = values[values.length - 1];
      return lastTrial.is_correct === 0; // Repeat if incorrect
    }
  };

// ============================================
// Learning Trials Timeline
// ============================================
// Create timeline with all 8 learning trials
// Each trial will loop until correct (handled by loop_function above)
const learning_trial = {
    timeline: [training_trial_template],
    timeline_variables: training_trials_data,
    randomize_order: false // Show in order, matching PsychoPy
};

// ============================================
// Phase X: Debrief Questionnaire
// ============================================
const debrief_trial = {
  type: jsPsychSurveyHtmlForm,
  html: `
    <div class="instruction-text">
      <h2>Debriefing</h2>
      <p>Thank you for completing the experiment! Please answer the following questions:</p>
      
      <div class="survey-question">
        <label for="debrief-purpose">What did you think this experiment was about?</label>
        <textarea id="debrief-purpose" name="debrief-purpose" required></textarea>
      </div>
      
      <div class="survey-question">
        <label for="debrief-comments">Do you have any comments or feedback about the experiment?</label>
        <textarea id="debrief-comments" name="debrief-comments"></textarea>
      </div>
    </div>
  `,
  data: {
    trial_type: 'debrief'
  }
};

// ============================================
// Phase 5: Prolific Completion Code Page
// ============================================
const completion_code_trial = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: `
    <div class="centered-content">
      <h2>Experiment Complete!</h2>
      <p class="completion-instructions">Please copy the code below and return to Prolific to complete the study.</p>
      <div class="completion-code">${EXPERIMENT_CONFIG.PROLIFIC_COMPLETION_CODE}</div>
      <p class="completion-instructions">Thank you for your participation!</p>
      <p class="prompt-text">Press any key to close this page</p>
    </div>
  `,
  choices: 'ALL_KEYS',
  data: {
    trial_type: 'completion_code',
    completion_code: EXPERIMENT_CONFIG.PROLIFIC_COMPLETION_CODE
  },
  on_finish: function() {
    // Attempt to save data to DataPipe if endpoint is configured
    if (EXPERIMENT_CONFIG.DATAPIPE_ENDPOINT) {
      saveToDataPipe();
    }
  }
};

// ============================================
// Phase 6: Data Collection Functions
// ============================================

// Function to save data to DataPipe (placeholder for now)
function saveToDataPipe() {
  // TODO: Implement DataPipe API call when endpoint is available
  const data = jsPsych.data.get().csv();
  console.log('Data to be sent to DataPipe:', data);
  
  // Placeholder for future implementation:
  // fetch(EXPERIMENT_CONFIG.DATAPIPE_ENDPOINT, {
  //   method: 'POST',
  //   headers: {
  //     'Content-Type': 'application/json',
  //   },
  //   body: JSON.stringify({
  //     participant_id: participantID,
  //     data: data
  //   })
  // })
  // .then(response => response.json())
  // .then(data => console.log('Data saved to DataPipe:', data))
  // .catch(error => console.error('Error saving to DataPipe:', error));
}

// ============================================
// Phase 7: Main Experiment Timeline
// ============================================

const timeline = [
  consent_trial,
  introduction_trial,
  exposure_instructions_trial,
  exposure_trials, // All 8 exposure trials
  learning_instructions_trial,
  learning_trial,
  debrief_trial,
  completion_code_trial
];

// Start the experiment
jsPsych.run(timeline);

