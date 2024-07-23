console.log('Content script is running');

// Listen for messages from the background script
browser.runtime.onMessage.addListener((message) => {
    if (message.action === 'showUrlWarning') {
        console.log('Phishing URL alert received from background script');
        showUrlWarningPopup(message.url);
    }
});

// URL popup warning
function showUrlWarningPopup(phishingUrl) {
    if (confirm("Phishing attempt detected! Do you want to continue to this URL?")) {
        window.location.href = phishingUrl;
    } else {
        window.history.back();
    }
}
