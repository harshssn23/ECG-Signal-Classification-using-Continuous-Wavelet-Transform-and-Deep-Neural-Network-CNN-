const sign_in_btn = document.querySelector("#sign-in-btn");
const sign_up_btn = document.querySelector("#sign-up-btn");
const container = document.querySelector(".container");

const loginBtn = document.querySelector("#login-btn");
const signupBtn = document.querySelector("#signup-btn");

sign_up_btn.addEventListener('click', () => {
    container.classList.add("sign-up-mode");
});

sign_in_btn.addEventListener('click', () => {
    container.classList.remove("sign-up-mode");
});

signupBtn.addEventListener('click', () => {
    const username = document.querySelector("#sign-up-username").value;
    const email = document.querySelector("#sign-up-email").value;
    const password = document.querySelector("#sign-up-password").value;

    if (username && email && password) {
        const users = JSON.parse(localStorage.getItem('users')) || {};
        if (users[username]) {
            alert('Username already exists!');
        } else {
            users[username] = { email, password };
            localStorage.setItem('users', JSON.stringify(users));
            alert('Sign Up Successful! You can now sign in.');
            sendWelcomeEmail(email, username); // Call function to send email
            container.classList.remove("sign-up-mode");
        }
    } else {
        alert('Please fill all fields!');
    }
});

loginBtn.addEventListener('click', () => {
    const username = document.querySelector("#sign-in-username").value;
    const password = document.querySelector("#sign-in-password").value;

    const users = JSON.parse(localStorage.getItem('users')) || {};

    if (users[username] && users[username].password === password) {
        alert('Login Successful!');
        window.location.href = "http://127.0.0.1:5000/";
    } else {
        alert('Invalid Username or Password!');
    }
});

function sendWelcomeEmail(email, username) {
    emailjs.send("YOUR_SERVICE_ID", "YOUR_TEMPLATE_ID", {
        to_email: email,
        to_name: username
    })
    .then((response) => {
        console.log('SUCCESS!', response.status, response.text);
    }, (error) => {
        console.log('FAILED...', error);
    });
}
