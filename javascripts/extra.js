// Custom JavaScript for HNSW documentation

document.addEventListener('DOMContentLoaded', function() {
  // Add copy buttons to code blocks
  document.querySelectorAll('pre code').forEach(function(codeBlock) {
    const container = codeBlock.parentNode;
    const copyButton = document.createElement('button');
    copyButton.className = 'md-clipboard';
    copyButton.title = 'Copy to clipboard';
    copyButton.innerHTML = '<span class="md-clipboard__message"></span>';
    
    copyButton.addEventListener('click', function() {
      const code = codeBlock.textContent;
      navigator.clipboard.writeText(code).then(function() {
        const message = copyButton.querySelector('.md-clipboard__message');
        message.textContent = 'Copied';
        setTimeout(function() {
          message.textContent = '';
        }, 2000);
      });
    });
    
    if (container.classList.contains('highlight')) {
      container.appendChild(copyButton);
    }
  });
  
  // Add anchor links to headings
  document.querySelectorAll('h2, h3, h4, h5, h6').forEach(function(heading) {
    if (heading.id) {
      const anchor = document.createElement('a');
      anchor.className = 'headerlink';
      anchor.href = '#' + heading.id;
      anchor.title = 'Permanent link';
      anchor.innerHTML = '¶';
      heading.appendChild(anchor);
    }
  });
  
  // Add version selector functionality
  const versionSelector = document.querySelector('.md-version-select');
  if (versionSelector) {
    versionSelector.addEventListener('change', function() {
      window.location.href = this.value;
    });
  }
  
  // Add search highlighting
  const searchParams = new URLSearchParams(window.location.search);
  const query = searchParams.get('q');
  if (query) {
    const words = query.split(/\s+/).filter(Boolean);
    const content = document.querySelector('.md-content');
    
    if (content && words.length) {
      const walker = document.createTreeWalker(
        content,
        NodeFilter.SHOW_TEXT,
        null,
        false
      );
      
      const nodesToHighlight = [];
      let node;
      
      while (node = walker.nextNode()) {
        const parent = node.parentNode;
        if (parent.tagName !== 'SCRIPT' && parent.tagName !== 'STYLE') {
          for (const word of words) {
            if (node.textContent.toLowerCase().includes(word.toLowerCase())) {
              nodesToHighlight.push(node);
              break;
            }
          }
        }
      }
      
      for (const node of nodesToHighlight) {
        const text = node.textContent;
        const fragment = document.createDocumentFragment();
        
        for (const word of words) {
          const regex = new RegExp(`(${word})`, 'gi');
          const parts = text.split(regex);
          
          for (const part of parts) {
            if (part.toLowerCase() === word.toLowerCase()) {
              const mark = document.createElement('mark');
              mark.textContent = part;
              fragment.appendChild(mark);
            } else {
              fragment.appendChild(document.createTextNode(part));
            }
          }
        }
        
        node.parentNode.replaceChild(fragment, node);
      }
    }
  }
}); 