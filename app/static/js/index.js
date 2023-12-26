function onValidate(formObj){
    if (formObj.username.value.length== 0){
      window.alert("Please Enter Valid name");
      formObj.focus();
      return false;
    }

    if (formObj.email.value.length== 0){
      window.alert("Please Enter Valid Email");
      formObj.focus();
      return false;
    }

    if (formObj.password.value.length== 0){
      window.alert("Please Enter Valid Password");
      formObj.focus();
      return false;
    }

    if (formObj.phone.value.length== 0){
      window.alert("Please Enter a phone");
      formObj.focus();
      return false;
    }
    else if(formObj.phone.value.length > 10){
      window.alert("Phone has only " + formObj.phone.value.length + "  " + "Charecters");
      formObj.focus();
      return false;
    }

    if (formObj.address.value.length== 0){
      window.alert("Please Enter Valid Address");
      formObj.focus();
      return false;
    }
  }

function verifyUser(){  

  if ("{{ flash_message }}" == "True"){
    alert("Message")

  }
}

function show_hide() {
  var listItems = document.getElementById("list-items");
  listItems.style.display = (listItems.style.display === "none" || listItems.style.display === "") ? "block" : "none";
}


function openPopup() {
  // Get the popup element
  var popup = document.getElementById("popup");

  // Display the popup
  popup.style.display = "block";
}

function closePopup() {
  // Get the popup element
  var popup = document.getElementById("popup");

  // Hide the popup
  popup.style.display = "none";
}
