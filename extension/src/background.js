browser.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('Background script received message:', message);
    
    if (message.action === 'checkEmail' && message.content) {
      console.log('Sending email content to backend for analysis...');
      
      /* fetch('http://localhost:5000/detect_phishing_email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email_content: message.content })
      })
      .then(response => response.json())
      .then(data => {
        console.log('Backend response:', data);
        if (true) {
          console.log('Phishing detected, notifying content script...');
          browser.tabs.sendMessage(sender.tab.id, { is_phishing: true });
        }
      })
      .catch(error => console.error('Error:', error)); */
    }
  });
  