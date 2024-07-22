console.log('Background script is running');

browser.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
    console.log('Background script received message:', message);
    
    if (message.action === 'checkEmail' && message.content) {
        console.log('Sending email content to backend for analysis...');
        
        try {
            const response = await fetch('http://localhost:5000/check/email', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email_content: message.content, sender: message.sender })
            });

            console.log('Received response from backend:', response);
            
            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Parsed backend response:', data);

            if (data.is_phishing) {
                console.log('Phishing detected, notifying content script...');
                browser.tabs.sendMessage(sender.tab.id, { action: 'showAlert', is_phishing: true });
            } else {
                console.log('No phishing detected.');
            }

        } catch (error) {
            console.error('Fetch error:', error);
        }
    }
});

browser.runtime.onInstalled.addListener(() => {
  console.log('Extension installed or updated.');
});
