// Compass HTML/CSS Functions

/**
 * Creates HTML for a compass (circle with arrow/hand) using CSS
 * @param {number} angle - Angle in degrees (PsychoPy system: 0 = up/12 o'clock, increases clockwise)
 * @returns {string} HTML string for the compass
 */
function createCompassHTML(angle) {
  // Convert PsychoPy angle (0 = up, clockwise) to CSS rotation
  // CSS rotate() uses 0 = right, clockwise, so we need: CSS_angle = angle - 90
  const cssAngle = angle - 90;
  
  return `
    <div class="compass-wrapper">
      <div class="compass-circle">
        <div class="compass-arrow" style="transform: rotate(${cssAngle}deg);"></div>
        <div class="compass-center"></div>
      </div>
    </div>
  `;
}

