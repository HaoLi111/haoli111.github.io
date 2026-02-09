document.addEventListener('DOMContentLoaded', (event) => {
  document.querySelectorAll('pre').forEach((pre) => {
    const button = document.createElement('button');
    button.className = 'copy-code-button';
    button.innerText = 'Copy';

    button.addEventListener('click', () => {
      const code = pre.querySelector('code') ? pre.querySelector('code').innerText : pre.innerText;
      
      navigator.clipboard.writeText(code).then(() => {
        button.innerText = 'Copied!';
        setTimeout(() => {
          button.innerText = 'Copy';
        }, 2000);
      }).catch(err => {
        console.error('Failed to copy: ', err);
      });
    });

    // Ensure parent is relative for absolute positioning of button
    if (getComputedStyle(pre).position === 'static') {
        pre.style.position = 'relative';
    }
    
    pre.appendChild(button);
  });
});
