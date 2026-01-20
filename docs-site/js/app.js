// ML Documentation Site JavaScript

class DocumentationSite {
    constructor() {
        this.currentPage = 'home';
        this.contentElement = document.getElementById('markdown-content');
        this.searchInitialized = false;
        this.init();
    }

    init() {
        // Configure MathJax
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                processEnvironments: true
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
            }
        };

        // Load initial page
        this.loadPage('home');
        
        // Handle browser back/forward
        window.addEventListener('popstate', (event) => {
            if (event.state && event.state.page) {
                this.loadPage(event.state.page, false);
            }
        });

        // Add mobile menu toggle
        this.setupMobileMenu();
    }

    async loadPage(pageName, pushState = true) {
        try {
            // Show loading state
            this.showLoading();
            
            // Update navigation
            this.updateNavigation(pageName);
            
            // Fetch markdown content
            // Use relative path - works for both local and GitHub Pages
            const pagePath = `pages/${pageName}.md`;
            const response = await fetch(pagePath);
            if (!response.ok) {
                throw new Error(`Failed to load page: ${response.status} - ${pagePath}`);
            }
            
            const markdown = await response.text();
            return this.processMarkdown(markdown, pageName, pushState);
            
        } catch (error) {
            console.error('Error loading page:', error);
            this.showError(`Failed to load page: ${pageName}`);
        }
    }

    processMarkdown(markdown, pageName, pushState) {
        // Convert markdown to HTML
        const html = marked.parse(markdown);
        
        // Update content
        this.contentElement.innerHTML = html;
        
        // Process math equations
        if (window.MathJax) {
            MathJax.typesetPromise([this.contentElement]).catch((err) => {
                console.warn('MathJax rendering error:', err);
            });
        }
        
        // Update browser history
        if (pushState) {
            const title = this.getPageTitle(pageName);
            history.pushState({ page: pageName }, title, `#${pageName}`);
            document.title = `${title} - ML Bootcamp Documentation`;
        }
        
        // Scroll to top
        window.scrollTo(0, 0);
        
        // Update current page
        this.currentPage = pageName;
        
        // Add syntax highlighting if available
        this.highlightCode();
        
        // Process tables for better styling
        this.enhanceTables();
        
        // Add copy buttons to code blocks
        this.addCopyButtons();
        
        // Setup search functionality (only once)
        if (!this.searchInitialized) {
            this.setupSearch();
            this.searchInitialized = true;
        }
            
        } catch (error) {
            console.error('Error loading page:', error);
            this.showError(`Failed to load page: ${pageName}`);
        }
    }

    showLoading() {
        this.contentElement.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <div class="loading-text">Loading content...</div>
            </div>
        `;
    }

    showError(message) {
        this.contentElement.innerHTML = `
            <div style="text-align: center; padding: 3rem; color: var(--danger-color);">
                <h2>⚠️ Error</h2>
                <p>${message}</p>
                <button onclick="app.loadPage('home')" style="
                    background: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 0.75rem 1.5rem;
                    border-radius: 0.5rem;
                    cursor: pointer;
                    margin-top: 1rem;
                ">Return Home</button>
            </div>
        `;
    }

    updateNavigation(pageName) {
        // Remove active class from all links
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.classList.remove('active');
        });
        
        // Add active class to current page link
        const currentLink = document.querySelector(`[onclick="loadPage('${pageName}')"]`);
        if (currentLink) {
            currentLink.classList.add('active');
        }
    }

    getPageTitle(pageName) {
        const titles = {
            'home': 'Home',
            'linear-regression': 'Linear Regression',
            'logistic-regression': 'Logistic Regression',
            'neural-networks': 'Neural Networks',
            'advanced-neural-networks': 'Advanced Neural Networks',
            'backpropagation': 'Backpropagation',
            'about': 'About'
        };
        return titles[pageName] || 'Documentation';
    }

    highlightCode() {
        // Basic syntax highlighting for Python code blocks
        const codeBlocks = this.contentElement.querySelectorAll('pre code');
        codeBlocks.forEach(block => {
            if (!block.classList.contains('highlighted')) {
                this.applyBasicHighlighting(block);
                block.classList.add('highlighted');
            }
        });
    }

    applyBasicHighlighting(block) {
        let html = block.innerHTML;
        
        // Python keywords
        const keywords = ['def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 'return', 'try', 'except', 'with', 'as'];
        keywords.forEach(keyword => {
            const regex = new RegExp(`\\b${keyword}\\b`, 'g');
            html = html.replace(regex, `<span style="color: #8b5cf6; font-weight: bold;">${keyword}</span>`);
        });
        
        // Strings
        html = html.replace(/(["'])((?:\\.|(?!\1)[^\\])*?)\1/g, '<span style="color: #10b981;">$1$2$1</span>');
        
        // Comments
        html = html.replace(/(#.*$)/gm, '<span style="color: #6b7280; font-style: italic;">$1</span>');
        
        // Numbers
        html = html.replace(/\b(\d+\.?\d*)\b/g, '<span style="color: #f59e0b;">$1</span>');
        
        block.innerHTML = html;
    }

    enhanceTables() {
        const tables = this.contentElement.querySelectorAll('table');
        tables.forEach(table => {
            // Add status badges for table cells containing "Complete", "Progress", etc.
            const cells = table.querySelectorAll('td');
            cells.forEach(cell => {
                const text = cell.textContent.trim();
                if (text === '✅ Complete') {
                    cell.innerHTML = '<span class="status-complete">✅ Complete</span>';
                } else if (text.includes('Progress')) {
                    cell.innerHTML = '<span class="status-progress">🔄 In Progress</span>';
                }
            });
        });
    }

    setupMobileMenu() {
        // Add mobile menu toggle button
        const sidebar = document.querySelector('.sidebar');
        const content = document.querySelector('.content');
        
        // Create mobile menu button
        const menuButton = document.createElement('button');
        menuButton.innerHTML = '☰';
        menuButton.style.cssText = `
            position: fixed;
            top: 1rem;
            left: 1rem;
            z-index: 1001;
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 0.75rem;
            border-radius: 0.5rem;
            cursor: pointer;
            display: none;
            font-size: 1.25rem;
        `;
        
        document.body.appendChild(menuButton);
        
        // Show menu button on mobile
        const checkMobile = () => {
            if (window.innerWidth <= 768) {
                menuButton.style.display = 'block';
            } else {
                menuButton.style.display = 'none';
                sidebar.classList.remove('open');
            }
        };
        
        window.addEventListener('resize', checkMobile);
        checkMobile();
        
        // Toggle mobile menu
        menuButton.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });
        
        // Close menu when clicking outside
        content.addEventListener('click', () => {
            sidebar.classList.remove('open');
        });
    }

    addCopyButtons() {
        const codeBlocks = this.contentElement.querySelectorAll('pre');
        codeBlocks.forEach((preBlock) => {
            // Skip if button already exists
            if (preBlock.querySelector('.copy-code-btn')) return;
            
            const button = document.createElement('button');
            button.className = 'copy-code-btn';
            button.textContent = 'Copy';
            button.setAttribute('aria-label', 'Copy code to clipboard');
            
            button.addEventListener('click', async () => {
                const code = preBlock.querySelector('code')?.textContent || preBlock.textContent;
                try {
                    await navigator.clipboard.writeText(code);
                    button.textContent = 'Copied!';
                    button.classList.add('copied');
                    setTimeout(() => {
                        button.textContent = 'Copy';
                        button.classList.remove('copied');
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy code:', err);
                    button.textContent = 'Failed';
                    setTimeout(() => {
                        button.textContent = 'Copy';
                    }, 2000);
                }
            });
            
            preBlock.style.position = 'relative';
            preBlock.appendChild(button);
        });
    }

    setupSearch() {
        const searchInput = document.getElementById('search-input');
        const searchResults = document.getElementById('search-results');
        
        if (!searchInput || !searchResults) return;
        
        // Available pages for search
        const searchablePages = [
            { name: 'home', title: 'Home' },
            { name: 'linear-regression', title: 'Linear Regression' },
            { name: 'logistic-regression', title: 'Logistic Regression' },
            { name: 'neural-networks', title: 'Neural Networks' },
            { name: 'advanced-neural-networks', title: 'Advanced Neural Networks' },
            { name: 'backpropagation', title: 'Backpropagation' },
            { name: 'about', title: 'About' }
        ];
        
        let searchTimeout;
        
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            const query = e.target.value.trim().toLowerCase();
            
            if (query.length < 2) {
                searchResults.classList.remove('show');
                searchResults.innerHTML = '';
                return;
            }
            
            searchTimeout = setTimeout(() => {
                const results = searchablePages.filter(page => 
                    page.title.toLowerCase().includes(query) ||
                    page.name.toLowerCase().includes(query)
                );
                
                if (results.length === 0) {
                    searchResults.innerHTML = '<div class="search-result-item">No results found</div>';
                } else {
                    searchResults.innerHTML = results.map(page => `
                        <div class="search-result-item" onclick="loadPage('${page.name}'); document.getElementById('search-input').value=''; document.getElementById('search-results').classList.remove('show');">
                            <strong>${this.highlightMatch(page.title, query)}</strong>
                            <div style="font-size: 0.875rem; color: var(--text-secondary); margin-top: 0.25rem;">
                                ${page.name}
                            </div>
                        </div>
                    `).join('');
                }
                
                searchResults.classList.add('show');
            }, 300);
        });
        
        // Close search on outside click
        document.addEventListener('click', (e) => {
            if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
                searchResults.classList.remove('show');
            }
        });
        
        // Close on Escape key
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                searchResults.classList.remove('show');
                searchInput.blur();
            }
        });
    }

    highlightMatch(text, query) {
        const regex = new RegExp(`(${query})`, 'gi');
        return text.replace(regex, '<mark style="background: rgba(37, 99, 235, 0.2); padding: 0.1em 0.2em; border-radius: 0.2em;">$1</mark>');
    }

    // Utility method for external access
    navigateTo(pageName) {
        this.loadPage(pageName);
    }
}

// Global function for onclick handlers
function loadPage(pageName) {
    if (window.app) {
        window.app.loadPage(pageName);
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DocumentationSite();
    
    // Handle initial hash navigation
    const hash = window.location.hash.substring(1);
    if (hash && hash !== 'home') {
        window.app.loadPage(hash);
    }
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DocumentationSite;
}