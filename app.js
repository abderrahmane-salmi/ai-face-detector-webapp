/**
 * AI Face Detector - Core Camera Stream Module
 * Handles asynchronous hardware streaming and DOM bindings.
 */

// 1. Capture references to the HTML elements we need to manipulate (DOM elements)
const webcamElement = document.getElementById('webcam');
const startButton = document.getElementById('start-camera-btn');

/**
 * Main initialization function to request permissions and stream webcam.
 */
async function setupWebcam() {
    // Check if the browser supports the required MediaDevices API
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Error: Your browser does not support webcam access (getUserMedia API).");
        return;
    }

    try {
        // Disable the button during the initialization sequence
        startButton.disabled = true;
        startButton.innerText = "Connecting...";

        // Request permission and capture raw hardware stream pipeline
        // This opens a browser prompt. Execution pauses here until the user accepts or denies.
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user" // Prioritize front-facing selfie camera
            },
            audio: false // We do not need audio for face detection
        });

        // 2. Bind the active camera stream pipeline directly to the video element's source object
        webcamElement.srcObject = stream;

        // 3. Wait until the video metadata has fully loaded and is ready to execute
        // We wrap this lifecycle event in a Promise to safely block downstream actions until the feed is hot.
        await new Promise((resolve) => {
            webcamElement.onloadedmetadata = () => {
                // The browser now knows the exact resolution and attributes of the incoming video feed
                resolve();
            };
        });

        // Play the video stream inside the element
        await webcamElement.play();
        
        // Update UI states on success
        startButton.innerText = "Camera Active";
        console.log("Success: Camera hardware pipeline established and streaming.");

    } catch (error) {
        // 4. Robust Error Handling Pipeline
        console.error("Camera Init Error Structure:", error);
        
        // Reset button state so user can retry if it was a transient error
        startButton.disabled = false;
        startButton.innerText = "Start Camera";

        // Evaluate specific error signatures thrown by the browser API
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            alert("Permission Denied: Please allow webcam access in your browser settings to run face detection.");
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            alert("Hardware Error: No compatible camera device could be found on your system.");
        } else {
            alert(`Stream Initialization Failed: ${error.message}`);
        }
    }
}

// 5. Attach an Event Listener to the "Start Camera" button
// This tells the browser engine to invoke setupWebcam when the user clicks the button.
startButton.addEventListener('click', setupWebcam);