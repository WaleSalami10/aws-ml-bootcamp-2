# AWS ML Bootcamp Documentation Website

A modern, responsive documentation website showcasing machine learning implementations and AWS service integrations from my 12-week bootcamp journey.

## Features

✅ **Responsive Design** - Works on desktop and mobile
✅ **Markdown Support** - Renders your .md files beautifully
✅ **Math Rendering** - LaTeX equations with MathJax
✅ **Syntax Highlighting** - Code blocks with proper formatting
✅ **Easy Navigation** - Sidebar with organized topics
✅ **No Build Step** - Pure HTML/CSS/JS

## Quick Start

### Option 1: Open Locally
```bash
cd docs-site
open index.html  # macOS
# or
start index.html  # Windows
# or
xdg-open index.html  # Linux
```

### Option 2: Use Python Server
```bash
cd docs-site
python3 -m http.server 8000
# Visit http://localhost:8000
```

### Option 3: Use VS Code Live Server
1. Install "Live Server" extension in VS Code
2. Right-click `index.html`
3. Select "Open with Live Server"

## Structure

```
docs-site/
├── index.html          # Main HTML file
├── css/
│   └── style.css       # Styles
├── js/
│   └── app.js          # JavaScript logic
└── README.md           # This file
```

## Adding New Pages

Edit `js/app.js` and add your markdown file path:

```javascript
case 'your-new-page':
    content = await loadMarkdownFile('../path/to/your/file.md');
    break;
```

Then add a link in `index.html`:

```html
<li><a href="#" onclick="loadPage('your-new-page')">📄 Your Page</a></li>
```

## Customization

### Colors
Edit `css/style.css` to change the color scheme:
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Logo
Change the logo in `index.html`:
```html
<h2>📚 Your Logo</h2>
```

## Technologies Used

- **Marked.js** - Markdown parser
- **MathJax** - Math rendering
- **Vanilla JavaScript** - No frameworks
- **CSS Grid/Flexbox** - Modern layouts

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## License

Free to use and modify for your projects!
