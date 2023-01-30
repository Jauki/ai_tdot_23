let videoStreamDisplay = null;

let websocket;

function connect() {
    websocket = new WebSocket("ws://localhost:5678/");
}

function subscribe() {
    let message = {
        "channel": "video_stream",
        "type": "subscribe"
    }
    websocket.send(JSON.stringify(message));
    console.log('subscribed');
}

function receive(data) {
    console.log("received frame");
    videoStreamDisplay.src = `data:image/png;base64,${data.payload.frame}`;
}

function send_control(x, y) {
    let message = {
        "channel": "control",
        "type": "publish",
        "x": x,
        "y": y,
    }
    websocket.send(JSON.stringify(message));
    console.log('sent control message');
}

function initWebsocket() {
    connect();
    websocket.onopen = () => {
        console.log('connected!');
        subscribe();
    }
    websocket.addEventListener("message", (message) => {
        receive(JSON.parse(message.data));
    })
}

window.addEventListener("DOMContentLoaded", () => {
    videoStreamDisplay = document.getElementById("videoStreamDisplay");
    initWebsocket();
});