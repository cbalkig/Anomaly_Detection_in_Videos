const s = document.getElementById('anomalyDetector');
const sourceVideo = s.getAttribute("data-source");
const uploadWidth = s.getAttribute("data-uploadWidth") || 640;
const apiServer = s.getAttribute("data-apiServer") || window.location.origin + '/recordingImage';

uuid = uuidv4();
console.info(uuid);
count = 0;
recording = null;

v = document.getElementById(sourceVideo);

let imageCanvas = document.createElement('canvas');
let imageCtx = imageCanvas.getContext("2d");

let drawCanvas = document.createElement('canvas');
document.body.appendChild(drawCanvas);
let drawCtx = drawCanvas.getContext("2d");

//Add file blob to a form and post
function postFile(file) {
    //Set options as form data
    let formdata = new FormData();
    formdata.append("image", file);
    formdata.append("uuid", uuid);
    formdata.append("count", ++count);

    let xhr = new XMLHttpRequest();
    xhr.open('POST', apiServer, true);
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

//Start object detection
function startAnomalyDetection() {
    if(!recording) {
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
    imageCanvas.toBlob(postFile, 'image/tif');
}

$(document).ready(function(){
    v.onplaying = () => {
        console.log("video playing");
        isPlaying = true;
        $('#startRecording').show();
    };

    $('#startRecording').click(function(){
        uuid = uuidv4();
        count = 0;
        console.info(uuid);
        recording = setInterval(startAnomalyDetection, 100);
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

