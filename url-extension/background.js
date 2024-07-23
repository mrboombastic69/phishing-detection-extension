console.log('Background script is running');

browser.webRequest.onBeforeRequest.addListener(
    async (details) => {
        const url = details.url;
        console.log(`Checking URL: ${url}`);

        try {
            const response = await fetch('http://localhost:5000/check/url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url })
            });

            console.log('Received URL response from backend:', response);

            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Parsed URL backend response:', data);

            if (data.is_phishing) {
                console.log('Phishing URL detected, blocking access...');
                // Notify the content script to show the warning popup
                await browser.tabs.sendMessage(details.tabId, { action: 'showUrlWarning', url: details.url });
                return { cancel: true };
            } else {
                console.log('No phishing URL detected.');
            }

        } catch (error) {
            console.error('Fetch error for URL:', error);
            return { cancel: true };
        }
    },
    { urls: ["<all_urls>"] },
    ["blocking"]
);

browser.runtime.onInstalled.addListener(() => {
    console.log('Extension installed or updated.');
});
