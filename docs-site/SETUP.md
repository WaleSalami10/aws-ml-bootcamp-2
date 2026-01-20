# Documentation Website Setup Guide

## 🚀 Quick Start Options

### Option 1: Local Development (Recommended)
```bash
cd docs-site
./deploy.sh
```
Visit: http://localhost:8080

### Option 2: Python Server
```bash
cd docs-site
python3 -m http.server 8080
```

### Option 3: VS Code Live Server
1. Install "Live Server" extension
2. Right-click `index.html` → "Open with Live Server"

## 🌐 GitHub Pages Deployment

### Automatic Deployment
1. Push your code to GitHub
2. Go to repository Settings → Pages
3. Select source: "Deploy from a branch"
4. Choose branch: `main` or `gh-pages`
5. Folder: `/docs-site` or `/` (if docs-site is root)

### Manual Setup
```bash
# Create gh-pages branch
git checkout -b gh-pages
git add docs-site/*
git commit -m "Deploy documentation site"
git push origin gh-pages
```

## 📝 Adding New Content

### 1. Create New Page
```bash
# Create new markdown file
touch docs-site/pages/new-algorithm.md
```

### 2. Add Navigation Link
Edit `docs-site/index.html`:
```html
<li><a href="#" onclick="loadPage('new-algorithm')">🔬 New Algorithm</a></li>
```

### 3. Write Content
Use markdown with LaTeX math support:
```markdown
# New Algorithm

Mathematical formula:
$$f(x) = ax^2 + bx + c$$

Code example:
```python
def new_algorithm(x, y):
    return x + y
```

## 🎨 Customization

### Colors & Branding
Edit `docs-site/css/style.css`:
```css
:root {
    --primary-color: #your-color;
    --secondary-color: #your-secondary;
}
```

### Logo & Title
Edit `docs-site/index.html`:
```html
<div class="logo">
    <h2>🎯 Your Project Name</h2>
</div>
```

## 📱 Mobile Optimization

The site is fully responsive with:
- Collapsible sidebar navigation
- Touch-friendly interface
- Optimized typography for mobile reading
- Fast loading on all devices

## 🔧 Advanced Features

### Math Rendering
- LaTeX equations with MathJax
- Inline: `$equation$`
- Display: `$$equation$$`

### Code Highlighting
- Automatic Python syntax highlighting
- Support for multiple languages
- Copy-to-clipboard functionality

### Performance
- Lazy loading for large content
- Optimized CSS and JavaScript
- Minimal external dependencies

## 🐛 Troubleshooting

### Math Not Rendering
- Check MathJax CDN connection
- Verify LaTeX syntax
- Clear browser cache

### Pages Not Loading
- Ensure markdown files are in `pages/` directory
- Check file naming matches navigation links
- Verify HTTP server is running

### Mobile Issues
- Test responsive breakpoints
- Check touch event handling
- Verify viewport meta tag

## 📊 Analytics (Optional)

Add Google Analytics to `index.html`:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

## 🔒 Security

- No server-side processing required
- Static files only
- Safe for GitHub Pages
- No sensitive data exposure

---

**Need Help?** Check the browser console for error messages or create an issue in the repository.