// Experiment Configuration
const EXPERIMENT_CONFIG = {
  // Prolific completion code (same for all participants)
  // TODO: Replace with actual code provided by Prolific
  PROLIFIC_COMPLETION_CODE: 'PLACEHOLDER_CODE',
  
  // DataPipe endpoint (to be configured later)
  // TODO: Replace with actual DataPipe endpoint URL
  DATAPIPE_ENDPOINT: null,
  
  // Experiment settings
  EXPERIMENT_NAME: 'elves_treasure_hunt',
  EXPERIMENT_VERSION: '1.0.0',
  
  // Timing constants (in milliseconds)
  ITI: 500, // Inter-trial interval
  FEEDBACK_TIME: 1000,
  
  // Other constants can be added here as needed
};

// Generate participant ID from URL parameter or create random one
function getParticipantID() {
  const urlParams = new URLSearchParams(window.location.search);
  const participantID = urlParams.get('participant_id') || 
                       urlParams.get('PROLIFIC_PID') ||
                       'participant_' + Math.random().toString(36).substr(2, 9);
  return participantID;
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { EXPERIMENT_CONFIG, getParticipantID };
}

