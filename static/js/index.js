$(document).ready(function () {
  setMultiSelect();

  htmx.on('#file_upload_form', 'htmx:xhr:progress', function (evt) {
    let percentage = evt.detail.loaded / evt.detail.total * 100
    htmx.find('#progress').setAttribute('aria-valuenow', percentage.toFixed(0))
    htmx.find('#progress').setAttribute('style', 'width:' + percentage.toFixed(0) + '%')
    htmx.find('#progress').innerHTML = percentage.toFixed(0) + '%'
  });
  htmx.onLoad(function (elt) {
    setMultiSelect();
  })
});

function setMultiSelect() {
  $('#selected_las').select2({
    multiple: true,
    placeholder: 'Click here to select files',
    closeOnSelect: false,
    theme: "bootstrap-5",
    //   {#selectionCssClass: "select2--small", // For Select2 v4.1#}
    //   {#dropdownCssClass: "select2--small",#}
  });
  $('#selected_logs').select2({
    dropdownParent: $("#las_preview_modal"),
    multiple: true,
    placeholder: 'Click here to select files',
    closeOnSelect: false,
    theme: "bootstrap-5",
  });
}

function closeModal() {
  const container = document.getElementById("feedback-modals-here")
  const backdrop = document.getElementById("feedback-modal-backdrop")
  const modal = document.getElementById("feedback-modal")

  modal.classList.remove("show")
  backdrop.classList.remove("show")

  setTimeout(function () {
    container.removeChild(backdrop)
    container.removeChild(modal)
  }, 200)
}