const navHomeBtn = document.querySelector("#nav-home");
const navTranslteBtn = document.querySelector("#nav-translate");
const navSettingsBtn = document.querySelector("#nav-settings");


//bottom nav
navHomeBtn.addEventListener("click", function() {
    window.location = '/settings'
})
navTranslteBtn.addEventListener("click", function() {
    window.location = '/translate'
})