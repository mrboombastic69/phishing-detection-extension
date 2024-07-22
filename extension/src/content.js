console.log('Content script is running');

function getEmailContent() {
  let emailContentElement = document.querySelector('.ii.gt');
  if (emailContentElement) {
    console.log('Email content extracted:', emailContentElement.innerText);
    return emailContentElement.innerText;
  }
  console.log('No email content found.');
  return '';
}

function getEmailSender() {
  let emailSenderElement = document.querySelector('.gD'); // Selector for the sender's email element
  if (emailSenderElement) {
    console.log('Email sender extracted:', emailSenderElement.getAttribute('email'));
    return emailSenderElement.getAttribute('email');
  }
  console.log('No email sender found.');
  return '';
}

function sendEmailContent() {
  const emailContent = getEmailContent();
  const emailSender = getEmailSender();
  
  if (emailContent && emailContent !== previousEmailContent) {
    const emailData = {
      action: 'checkEmail',
      content: emailContent,
      sender: emailSender,
    };
    browser.runtime.sendMessage(emailData);
    console.log('Sent email data to background script:', emailData);
    previousEmailContent = emailContent; // Update the previously sent content
    emailContentSent = true; // Set flag to true after sending
  }
}


// Debounce function to limit how often sendEmailContent is called
function debounce(func, wait) {
  let timeout;
  return function() {
    const context = this, args = arguments;
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(context, args), wait);
  };
}

// Create a debounced version of sendEmailContent
const debouncedSendEmailContent = debounce(sendEmailContent, 2000);

// MutationObserver to detect changes in the email content area
const observer = new MutationObserver((mutations) => {
  mutations.forEach((mutation) => {
    if (mutation.addedNodes.length > 0) {
      debouncedSendEmailContent();
    }
  });
});

// Observe changes in the Gmail email content area
const emailContentArea = document.querySelector('body');
if (emailContentArea) {
  observer.observe(emailContentArea, { childList: true, subtree: true });
} else {
  console.log('Email content area not found.');
}

