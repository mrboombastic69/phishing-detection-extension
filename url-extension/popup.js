document.getElementById('goBack').addEventListener('click', () => {
  window.history.back();
});

document.getElementById('proceed').addEventListener('click', () => {
  const urlParams = new URLSearchParams(window.location.search);
  const url = urlParams.get('url');
  window.location.href = url;
});
