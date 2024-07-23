async function checkUrl(url) {
    const response = await fetch('http://localhost:5000/check/url', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: url })
    });
    const data = await response.json();
    return data.is_phishing;
  }
  
  browser.webRequest.onBeforeRequest.addListener(
    async (details) => {
      if (details.type === "main_frame") {
        const isPhishing = await checkUrl(details.url);
        if (isPhishing) {
          console.log(isPhishing)
          browser.tabs.create({
            url: browser.extension.getURL("popup.html")
          });
        }
      }
    },
    { urls: ["<all_urls>"] },
    ["blocking"]
  );