let videoStreamDisplay = null;
let faceRecognitionResult = null;
let websocket;

function connect() {
    websocket = new WebSocket("ws://localhost:5678/");
}

function subscribe(channel) {
    let message = {
        "channel": channel,
        "type": "subscribe"
    }
    websocket.send(JSON.stringify(message));
    console.log('subscribed');
}

function receive(message) {
    switch (message.channel) {
        case "video_stream": {
            handle_video_stream(message);
            break;
        }
        case "face_recognition_result": {
            console.log(`received message: ${JSON.stringify(message)}`)
            handle_face_recognition(message);
            break;
        }
    }
}

function handle_face_recognition(message) {
    let result = message.payload.result;
    let certainty = parseFloat(result.certainty) * 100;
    faceRecognitionResult.textContent = `${result.label} (certainty: ${parseFloat(certainty).toFixed(2)} %)`;
}

function handle_video_stream(message) {
    videoStreamDisplay.src = `data:image/png;base64,${message.payload.frame}`;
}

function initWebsocket() {
    connect();
    websocket.onopen = () => {
        console.log('connected!');
        subscribe("video_stream");
        subscribe("face_recognition_result");
    }
    websocket.addEventListener("message", (message) => {
        receive(JSON.parse(message.data));
    })
}

window.addEventListener("DOMContentLoaded", () => {
    videoStreamDisplay = document.getElementById("videoStreamDisplay");
    faceRecognitionResult = document.getElementById("faceRecognitionResult");
    initWebsocket();
});