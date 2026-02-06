// MDIMG QA — Upload page JS
(function () {
    "use strict";

    const zone = document.getElementById("dropZone");
    const fileInput = document.getElementById("dicomFile");
    const fileLabel = document.getElementById("fileLabel");
    const form = document.getElementById("uploadForm");
    const submitBtn = document.getElementById("submitBtn");
    const spinner = document.getElementById("submitSpinner");
    const ALLOWED = [".dcm", ".dicom"];

    if (!zone) return;

    // Drag-and-drop handlers
    ["dragenter", "dragover"].forEach((evt) => {
        zone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            zone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((evt) => {
        zone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            zone.classList.remove("dragover");
        });
    });

    zone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files.length) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    });

    zone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) handleFileSelect(fileInput.files[0]);
    });

    function handleFileSelect(file) {
        const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
        if (!ALLOWED.includes(ext) && ext !== "") {
            fileLabel.textContent = "⚠️ Only .dcm / .dicom files allowed";
            fileLabel.classList.add("text-danger");
            submitBtn.disabled = true;
            return;
        }
        fileLabel.textContent = file.name + " (" + (file.size / 1024).toFixed(1) + " KB)";
        fileLabel.classList.remove("text-danger");
        submitBtn.disabled = false;
    }

    // Form submit — show spinner, disable button
    if (form) {
        form.addEventListener("submit", () => {
            submitBtn.disabled = true;
            if (spinner) spinner.classList.remove("d-none");
        });
    }
})();
