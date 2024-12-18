const navHomeBtn = document.querySelector("#nav-home");
const navTranslteBtn = document.querySelector("#nav-translate");
const navSettingsBtn = document.querySelector("#nav-settings");

const translateCard = document.querySelector("#translate-card");
const learnCard = document.querySelector("#learn-card");
const contributeCard = document.querySelector("#contribute-card");
const contactUsCard = document.querySelector("#contact-us-card");

//card nav
translateCard.addEventListener("click", function () {
  window.location = "./translate";
});
learnCard.addEventListener("click", function () {
  window.location = "./learn";
});
contributeCard.addEventListener("click", function () {
  window.location = "";
});
contactUsCard.addEventListener("click", function () {
  window.location = "./contact-us";
});

//bottom nav
navTranslteBtn.addEventListener("click", function () {
  window.location = "./translate";
});
navSettingsBtn.addEventListener("click", function () {
  navHomeBtn.classList.remove("active-nav-btn");
  navSettingsBtn.classList.add("active-nav-btn");
  window.location = "#";
});

const userPfp = document.getElementById("user-pfp");
const detailsModal = document.getElementById("detailsModal");
const detailsTextarea = document.getElementById("detailsTextarea");
const saveButton = document.getElementById("saveButton");

userPfp.addEventListener("click", () => {
  detailsModal.style.display = "block";

  const storedDetails = localStorage.getItem("details");
  if (storedDetails) {
    detailsTextarea.value = storedDetails;
  } else {
    localStorage.setItem(
      "details",
      `My name is Ethan Singhania \n
        I live in Bandra, Mumbai \n
        I study at Metro Institute of Technology (MIT) \n
        I have disability and I am Deaf and Dumb \n
        I am proficient in Indian Sign Language \n
        I am fluent in Hindi reading and writing \n
        My emergency phone number is 7777777777 \n
        My father's name is sasmita Singhania \n
        I am 21 years old.`
    );
  }
});

saveButton.addEventListener("click", () => {
  const details = detailsTextarea.value;
  localStorage.setItem("details", details);
  detailsModal.style.display = "none";
});
