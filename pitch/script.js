/* ============================================================
   MotionGuru Pitch Deck — Navigation & Animations
   JS-driven snap: always lands on a slide, but allows rapid input.
   ============================================================ */
(function () {
  const slides = document.querySelectorAll('.slide');
  const nav = document.getElementById('slide-nav');
  const counter = document.getElementById('slide-counter');
  const bar = document.getElementById('progress-bar');

  // Build nav dots
  slides.forEach((_, i) => {
    const dot = document.createElement('button');
    dot.classList.add('dot');
    dot.setAttribute('aria-label', `Go to slide ${i + 1}`);
    dot.addEventListener('click', () => goToSlide(i));
    nav.appendChild(dot);
  });
  const dots = nav.querySelectorAll('.dot');

  // Intersection observer — highlights active slide & triggers animations
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const idx = [...slides].indexOf(entry.target);
        dots.forEach((d, i) => d.classList.toggle('active', i === idx));
        counter.textContent = `${idx + 1} / ${slides.length}`;
        bar.style.width = `${((idx + 1) / slides.length) * 100}%`;
        entry.target.querySelectorAll('.fade-up').forEach(el => el.classList.add('visible'));
      }
    });
  }, { threshold: 0.45 });

  slides.forEach(s => observer.observe(s));

  // ── Snap Navigation ──
  let currentIdx = 0;
  let navLock = false;
  const COOLDOWN = 300; // ms — short enough for rapid scrolling, long enough to finish animation

  function getCurrentSlide() {
    let best = 0, bestDist = Infinity;
    slides.forEach((s, i) => {
      const dist = Math.abs(s.getBoundingClientRect().top);
      if (dist < bestDist) { bestDist = dist; best = i; }
    });
    return best;
  }

  function goToSlide(idx) {
    if (idx < 0 || idx >= slides.length) return;
    if (navLock && idx === currentIdx) return;
    navLock = true;
    currentIdx = idx;
    slides[idx].scrollIntoView({ behavior: 'smooth' });
    setTimeout(() => { navLock = false; }, COOLDOWN);
  }

  // Keyboard: arrow keys
  document.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      e.preventDefault();
      goToSlide(getCurrentSlide() + 1);
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      goToSlide(getCurrentSlide() - 1);
    }
  });

  // Mouse wheel / trackpad: snap one slide per scroll gesture
  document.addEventListener('wheel', e => {
    e.preventDefault();
    if (e.deltaY > 0) {
      goToSlide(getCurrentSlide() + 1);
    } else if (e.deltaY < 0) {
      goToSlide(getCurrentSlide() - 1);
    }
  }, { passive: false });

  // Touch swipe support for mobile
  let touchStartY = 0;
  document.addEventListener('touchstart', e => {
    touchStartY = e.touches[0].clientY;
  }, { passive: true });

  document.addEventListener('touchend', e => {
    const delta = touchStartY - e.changedTouches[0].clientY;
    if (Math.abs(delta) > 50) { // minimum swipe distance
      goToSlide(getCurrentSlide() + (delta > 0 ? 1 : -1));
    }
  }, { passive: true });
})();
