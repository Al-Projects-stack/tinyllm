# Al Mujati — Portfolio Site

A dual-theme personal developer portfolio built with pure HTML, CSS, and JavaScript — no frameworks, no build tools.

## Themes

### Cyber Edition (`cyber/`)
- Dark terminal aesthetic with cyan accent (`#00AAFF`)
- Animated particle network canvas with mouse interaction
- Custom glowing cursor with trail
- Typewriter hero animation
- Scroll-reveal sections (About, Projects, Skills, Contact)
- Responsive layout

### Vista Edition (`vista/`)
- Full Windows Vista desktop simulation
- Animated boot sequence (plays once per session)
- CSS/SVG Aurora wallpaper (no external images)
- Aero glass windows with blur effects
- Draggable, resizable, minimizable, maximizable windows
- Taskbar with Start button, open-window pills, system tray clock
- Start menu with pinned apps and links
- Desktop icons with double-click to open
- Right-click context menu on desktop (Refresh, Change Theme, View Source, Properties)
- Recycle Bin easter egg with right-click context menu
- Mobile notice with link to Cyber Edition

## Usage

Open `index.html` in a browser. Choose a theme on first load — your choice is saved in localStorage for repeat visits.

```
portfolio/
├── index.html          # Theme selector
├── cyber/
│   └── index.html      # Cyber edition
├── vista/
│   └── index.html      # Vista edition
├── assets/
│   ├── css/
│   ├── js/
│   └── images/
└── README.md
```

## Technical Decisions

- **Zero dependencies** — no React, no libraries, no build step. Just a browser.
- **localStorage** — remembers theme choice so the selector is skipped on return visits.
- **sessionStorage** — Vista boot sequence plays once per session, not per visit.
- **Canvas particles** — limited to 80 particles with connection distance threshold for performance.
- **Aurora wallpaper** — pure CSS `radial-gradient` with `blur()` — no images loaded.
- **Aero glass** — CSS `backdrop-filter: blur(20px)` on semi-transparent backgrounds.
- **Dragging** — mousedown/mousemove/mouseup on title bars, constrained to viewport minus taskbar.
- **Z-index stacking** — dynamically incremented on focus; windows tracked in ordered array.

## Author

**Al Mujati** — almujati02@gmail.com
