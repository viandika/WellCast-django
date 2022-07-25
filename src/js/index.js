$(function () {
  setMultiSelect();

  htmx.on("#file_upload_form", "htmx:xhr:progress", function (evt) {
    let percentage = (evt.detail.loaded / evt.detail.total) * 100;
    htmx.find("#progress").setAttribute("aria-valuenow", percentage.toFixed(0));
    htmx
      .find("#progress")
      .setAttribute("style", "width:" + percentage.toFixed(0) + "%");
    htmx.find("#progress").innerHTML = percentage.toFixed(0) + "%";
  });
  htmx.onLoad(function () {
    setMultiSelect();
  });
});

function setMultiSelect() {
  $("#selected_las").select2({
    multiple: true,
    placeholder: "Click here to select files",
    closeOnSelect: false,
    theme: "bootstrap-5",
    //   {#selectionCssClass: "select2--small", // For Select2 v4.1#}
    //   {#dropdownCssClass: "select2--small",#}
  });
  $("#selected_logs").select2({
    dropdownParent: $("#preview-las-modal"),
    multiple: true,
    placeholder: "Click here to select logs",
    closeOnSelect: false,
    theme: "bootstrap-5",
  });
  $("#selected_log").select2({
    dropdownPosition: "below",
    multiple: true,
    placeholder: "Click here to select logs",
    closeOnSelect: false,
    theme: "bootstrap-5",
  });
}

window.closeModal = function closeModal() {
  const container = document.getElementById("feedback-modals-here");
  const backdrop = document.getElementById("feedback-modal-backdrop");
  const modal = document.getElementById("feedback-modal");

  modal.classList.remove("show");
  backdrop.classList.remove("show");

  setTimeout(function () {
    container.removeChild(backdrop);
    container.removeChild(modal);
  }, 200);
};
window.closePreviewModal = function closePreviewModal() {
  const container = document.getElementById("las-preview-train-modal-here");
  const backdrop = document.getElementById("preview-las-modal-backdrop");
  const modal = document.getElementById("preview-las-modal");

  modal.classList.remove("show");
  backdrop.classList.remove("show");

  setTimeout(function () {
    container.removeChild(backdrop);
    container.removeChild(modal);
  }, 200);
};

window.closePredPreviewModal = function closePredPreviewModal() {
  const container = document.getElementById("preview-las-modals-here");
  const backdrop = document.getElementById("preview-las-modal-backdrop");
  const modal = document.getElementById("preview-las-modal");

  modal.classList.remove("show");
  backdrop.classList.remove("show");

  setTimeout(function () {
    container.removeChild(backdrop);
    container.removeChild(modal);
  }, 200);
};
