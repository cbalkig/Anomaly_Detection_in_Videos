function gotDevices(deviceInfos) {
    let videoSource = undefined;
    for (let i = 0; i !== deviceInfos.length; ++i) {
        const deviceInfo = deviceInfos[i];
        if (deviceInfo.kind === 'videoinput' && deviceInfo.label == 'Back Camera') {
            videoSource = deviceInfo.deviceId;
        }
    }

    const constraints = {
        audio: false,
        video: {deviceId: videoSource ? {exact: videoSource} : undefined}
    }

    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {
            window.stream = stream;
            document.getElementById("myVideo").srcObject = stream;
            console.log("Got local user video");
        })
        .catch(err => {
            console.log('navigator.getUserMedia error: ', err)
        });
}

navigator.mediaDevices.enumerateDevices().then(gotDevices);


const s = document.getElementById('anomalyDetector');
const sourceVideo = s.getAttribute("data-source");
const uploadWidth = s.getAttribute("data-uploadWidth") || 640;
const apiServer = s.getAttribute("data-apiServer") || window.location.origin;

uuid = uuidv4();
console.info(uuid);
recordingCount = 0;
recording = null;
anomalyDetecting = null;
anomalyCount = 0;

v = document.getElementById(sourceVideo);

let imageCanvas = document.createElement('canvas');
let imageCtx = imageCanvas.getContext("2d");

let drawCanvas = document.createElement('canvas');
document.body.appendChild(drawCanvas);
let drawCtx = drawCanvas.getContext("2d");

//Add file blob to a form and post
function recordFile(file) {
    //Set options as form data
    let formdata = new FormData();
    formdata.append("image", file);
    formdata.append("uuid", uuid);
    formdata.append("count", ++recordingCount);

    let xhr = new XMLHttpRequest();
    xhr.open('POST', apiServer + "/recordingImage", true);
    xhr.onload = function () {
        if (this.status === 200) {
            console.info(this.response)
        }
        else {
            console.error(xhr);
        }
    };
    xhr.send(formdata);
}

function startRecording() {
    if(!recording) {
        return;
    }
    console.log("starting recording");

    //Set canvas sizes base don input video
    drawCanvas.width = v.videoWidth;
    drawCanvas.height = v.videoHeight;

    imageCanvas.width = uploadWidth;
    imageCanvas.height = uploadWidth * (v.videoHeight / v.videoWidth);

    //Some styles for the drawcanvas
    drawCtx.lineWidth = 4;
    drawCtx.strokeStyle = "cyan";
    drawCtx.font = "20px Verdana";
    drawCtx.fillStyle = "cyan";

    //Save and send the first image
    imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
    imageCanvas.toBlob(recordFile, 'image/tif');
}


function doAnomalyDetection(file) {
    //Set options as form data
    let formdata = new FormData();
    formdata.append("image", file);
    formdata.append("uuid", uuid);

    let xhr = new XMLHttpRequest();
    xhr.open('POST', apiServer + "/anomalyDetection", true);
    xhr.onload = function () {
        if (this.status === 200) {
            console.info(this.response)
            if (!anomalyDetecting) {
                $("#abnormal").hide();
                return
            }

            if ("1" === this.response) {
                $("#abnormal").show();
            }
            else {
                $("#abnormal").hide();
            }
        }
        else {
            console.error(xhr);
        }
    };
    xhr.send(formdata);
}

function startAnomalyDetection() {
    if(!anomalyDetecting) {
        return;
    }
    console.log("starting anomaly detection");

    //Set canvas sizes base don input video
    drawCanvas.width = v.videoWidth;
    drawCanvas.height = v.videoHeight;

    imageCanvas.width = uploadWidth;
    imageCanvas.height = uploadWidth * (v.videoHeight / v.videoWidth);

    //Some styles for the drawcanvas
    drawCtx.lineWidth = 4;
    drawCtx.strokeStyle = "cyan";
    drawCtx.font = "20px Verdana";
    drawCtx.fillStyle = "cyan";

    //Save and send the first image
    imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
    imageCanvas.toBlob(doAnomalyDetection, 'image/tif');
}

$(document).ready(function(){
    v.onplaying = () => {
        console.log("video playing");
        isPlaying = true;
        $('#startRecording').show();
        $('#anomalyDetection').show();
    };

    $('#anomalyDetection').click(function(){
        anomalyCount = 0;
        anomalyDetecting = setInterval(startAnomalyDetection, 100);
        $('#startRecording').hide();
        $('#anomalyDetection').hide();
        $('#stopAnomalyDetection').show();
    });

    $('#stopAnomalyDetection').click(function(){
        $("#abnormal").hide();
        $('#startRecording').show();
        $('#anomalyDetection').show();
        $('#stopAnomalyDetection').hide();
        clearInterval(anomalyDetecting);
        anomalyDetecting = null;
    });

    $('#startRecording').click(function(){
        uuid = uuidv4();
        recordingCount = 0;
        console.info(uuid);
        recording = setInterval(startRecording, 50);
        $('#stopRecording').show();
        $("#recording").show();
        $('#startRecording').hide();
    });

    $('#stopRecording').click(function(){
        $("#stopRecording").hide();
        $("#recording").hide();
        $('#startRecording').show();
        clearInterval(recording);
        recording = null;
    });
});

function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

