import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

function findDatasetWidget(node) {
    return node.widgets?.find((widget) => widget.name === "dataset_zip");
}

async function uploadZip(file) {
    const body = new FormData();
    body.append("image", file);
    body.append("type", "input");
    body.append("overwrite", "false");

    const response = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
    });

    if (!response.ok) {
        throw new Error(`ZIP upload failed: ${response.status}`);
    }

    return await response.json();
}

function setComboValue(widget, value) {
    widget.value = value;
    if (Array.isArray(widget.options?.values) && !widget.options.values.includes(value)) {
        widget.options.values.push(value);
        widget.options.values.sort((a, b) => String(a).localeCompare(String(b)));
    }
    if (typeof widget.callback === "function") {
        widget.callback(value);
    }
}

function addZipUploadButton(node, label) {
    const uploadWidget = node.addWidget("button", label, label, async () => {
        const widget = findDatasetWidget(node);
        if (!widget) {
            alert("dataset_zip widget not found.");
            return;
        }

        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".zip,application/zip";
        input.style.display = "none";

        input.onchange = async () => {
            const file = input.files?.[0];
            input.remove();
            if (!file) {
                return;
            }
            if (!file.name.toLowerCase().endsWith(".zip")) {
                alert("Please choose a .zip dataset file.");
                return;
            }

            const oldLabel = node.title;
            node.title = `${oldLabel} - uploading...`;
            try {
                const result = await uploadZip(file);
                const uploadedName = result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
                setComboValue(widget, uploadedName);
                node.setDirtyCanvas(true, true);
            } catch (error) {
                console.error(error);
                alert(error.message || "ZIP upload failed.");
            } finally {
                node.title = oldLabel;
            }
        };

        document.body.appendChild(input);
        input.click();
    });
    uploadWidget.serialize = false;
}

function addZipUploadButtons(node) {
    if (node.__runninghubZipUploadAdded) {
        return;
    }
    node.__runninghubZipUploadAdded = true;

    addZipUploadButton(node, "Upload ZIP | RHUB");
    addZipUploadButton(node, "Upload ZIP");
}

function makeOutputDownloadUrl(fileInfo) {
    const params = new URLSearchParams();
    params.set("filename", fileInfo.filename);
    params.set("type", fileInfo.type || "output");
    if (fileInfo.subfolder) {
        params.set("subfolder", fileInfo.subfolder);
    }
    return `/view?${params.toString()}`;
}

function downloadFile(fileInfo) {
    const url = makeOutputDownloadUrl(fileInfo);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = fileInfo.filename;
    anchor.style.display = "none";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
}

function addExportDownloadButton(node, label) {
    const downloadWidget = node.addWidget("button", label, label, () => {
        const fileInfo = node.__runninghubLatestDownload;
        if (!fileInfo) {
            alert("No exported ZIP yet. Run the workflow first.");
            return;
        }
        downloadFile(fileInfo);
    });
    downloadWidget.serialize = false;
}

function addExportDownloadButtons(node) {
    if (node.__runninghubExportButtonsAdded) {
        return;
    }
    node.__runninghubExportButtonsAdded = true;

    addExportDownloadButton(node, "Download ZIP | RHUB");
    addExportDownloadButton(node, "Download ZIP");
}

function rememberExportDownloads(node, message) {
    const downloads = message?.runninghub_downloads || message?.ui?.runninghub_downloads;
    if (!Array.isArray(downloads) || downloads.length === 0) {
        return;
    }
    node.__runninghubLatestDownload = downloads[0];
    node.title = "RunningHub LoRA Export - ready";
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "darkhub.LoadImageSetFromFolderNode.ZipUpload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const className = nodeData?.name || nodeData?.type || nodeType?.comfyClass;
        if (className === "LoadImageSetFromFolderNode") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);
                addZipUploadButtons(this);
            };
            return;
        }

        if (className === "RunningHubLoRAExport") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);
                addExportDownloadButtons(this);
            };

            const originalOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                originalOnExecuted?.apply(this, arguments);
                rememberExportDownloads(this, message);
            };
        }
    },
});
