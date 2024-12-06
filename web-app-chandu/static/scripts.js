document.getElementById('login-button').addEventListener('click', () => {
    fetch("/start-face-recognition")
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert(data.message); 
            window.location.href = "/eye-tracking"; 
        } else {
            alert("Error: " + data.error || data.message);
        }
    })
    .catch(error => console.error("Error:", error));
});
