/* ========================================
   QUANTUMTRADE PRO — Global Frontend Logic
   ======================================== */

document.addEventListener('DOMContentLoaded', () => {
    // 1. THEME SWITCHER LOGIC
    const themeToggle = document.getElementById('themeToggle');
    const body = document.body;

    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
        body.classList.add('light-theme');
        updateThemeIcon(true);
    }

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const isLight = body.classList.toggle('light-theme');
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
            updateThemeIcon(isLight);
        });
    }

    function updateThemeIcon(isLight) {
        const icon = document.querySelector('#themeToggle i');
        if (icon) {
            icon.className = isLight ? 'bi bi-brightness-high-fill' : 'bi bi-moon-stars-fill';
        }
    }

    // 2. MOBILE NAVIGATION LOGIC
    const mobileNavBtn = document.getElementById('mobileNavBtn');
    const navLinks = document.querySelector('.nav-links');

    if (mobileNavBtn && navLinks) {
        mobileNavBtn.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            const icon = mobileNavBtn.querySelector('i');
            if (icon) {
                icon.className = navLinks.classList.contains('active') ? 'bi bi-x-lg' : 'bi bi-list';
            }
        });
    }

    // 3. AUTO-HIDE FLASH MESSAGES
    const flashMessages = document.querySelectorAll('.flash-msg');
    flashMessages.forEach(msg => {
        setTimeout(() => {
            msg.style.opacity = '0';
            msg.style.transform = 'translateY(-10px)';
            msg.style.transition = 'all 0.5s ease';
            setTimeout(() => msg.remove(), 500);
        }, 5000);
    });
});
