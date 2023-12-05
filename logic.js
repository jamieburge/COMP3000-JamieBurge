const webcam = document.getElementById('webcam');

// Use navigator.mediaDevices.getUserMedia to access the webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        // Set the video element's source to the webcam stream
        webcam.srcObject = stream;
    })
    .catch((error) => {
        console.error('Error accessing webcam:', error);
    });