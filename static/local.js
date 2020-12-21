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