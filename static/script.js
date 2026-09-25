function startAnalysis() {

    document.getElementById("analysis").scrollIntoView({
        behavior: "smooth"
    });

}


function uploadDataset() {

    const fileInput =
        document.getElementById("datasetFile");

    const result =
        document.getElementById("result");


    if (fileInput.files.length === 0) {

        result.innerHTML =
            "<p>Please select a CSV file first.</p>";

        return;
    }


    const file = fileInput.files[0];


    if (!file.name.endsWith(".csv")) {

        result.innerHTML =
            "<p>Please upload a CSV file.</p>";

        return;
    }


    result.innerHTML =
        "<p>Dataset selected: "
        + file.name
        + "</p>";

}