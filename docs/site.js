(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Theme toggle: system → light → dark. Remembered per browser when storage is available.
  const root = document.documentElement;
  const themeButton = document.getElementById('theme');
  const store = { get() { try { return localStorage.getItem('ruvllm-theme'); } catch { return null; } },
    set(v) { try { v ? localStorage.setItem('ruvllm-theme', v) : localStorage.removeItem('ruvllm-theme'); } catch {} } };
  const applyTheme = (v) => { v ? root.setAttribute('data-theme', v) : root.removeAttribute('data-theme'); themeButton.textContent = v ? v[0].toUpperCase() + v.slice(1) : 'System'; };
  applyTheme(store.get());
  themeButton.addEventListener('click', () => {
    const next = { null: 'light', light: 'dark', dark: null }[root.getAttribute('data-theme')];
    store.set(next); applyTheme(next);
  });

  // Hero background: a field of "weights", a few of which light up.
  const hero = document.getElementById('hero-grid');
  for (let i = 0; i < 24 * 10; i++) {
    const cell = document.createElement('i');
    cell.style.animationDelay = `${(Math.sin(i * 12.9898) * 43758.5453 % 1 + 1) % 1 * 7}s`;
    hero.appendChild(cell);
  }

  // Scroll progress.
  const bar = document.getElementById('progress');
  const onScroll = () => {
    const max = document.documentElement.scrollHeight - innerHeight;
    bar.style.transform = `scaleX(${max > 0 ? scrollY / max : 0})`;
  };
  addEventListener('scroll', onScroll, { passive: true }); onScroll();

  // Dot grids built from the measured counts (order: categories laid out in sequence).
  const dot = (parent, cx, cy, r, cls, delay) => {
    const c = document.createElementNS(NS, 'circle');
    c.setAttribute('cx', cx); c.setAttribute('cy', cy); c.setAttribute('r', r);
    c.setAttribute('class', `dot ${cls}`);
    if (!reduced) c.style.transitionDelay = `${delay}ms`;
    parent.appendChild(c);
  };
  const grid = (id, parts) => {
    const g = document.getElementById(id);
    const classes = parts.flatMap(([cls, n]) => Array(n).fill(cls));
    classes.forEach((cls, i) => dot(g, 100 + (i % 16) * 26.5, 182 + Math.floor(i / 16) * 25, 9, cls, i * 5));
  };
  grid('grid-v3', [['pass', 123], ['gain', 50], ['', 19]]);
  grid('grid-v4', [['pass', 136], ['gain', 40], ['loss', 13], ['', 3]]);
  const lane = (id, y, cls) => {
    const g = document.getElementById(id);
    for (let i = 0; i < 256; i++) dot(g, 66 + (i % 32) * 15, y + Math.floor(i / 32) * 15, 5, cls, i * 7);
  };
  lane('lane-base', 132, 'base');
  lane('lane-adpt', 322, 'adpt');

  // Scrollytelling: the step nearest the viewport centre drives the sticky stage.
  const steps = [...document.querySelectorAll('.step')];
  const scenes = [...document.querySelectorAll('.scene')];
  let current = -1;
  const show = (n) => {
    if (n === current) return;
    current = n;
    steps.forEach((s) => s.classList.toggle('is-active', +s.dataset.scene === n));
    scenes.forEach((s) => s.classList.toggle('on', +s.dataset.scene === n));
  };
  show(0);
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) show(+e.target.dataset.scene); });
  }, { rootMargin: matchMedia('(max-width: 860px)').matches ? '-62% 0px -12% 0px' : '-45% 0px -45% 0px' });
  steps.forEach((s) => io.observe(s));

  // Copy buttons.
  document.querySelectorAll('.code').forEach((block) => {
    const button = block.querySelector('.copy');
    const code = block.querySelector('code');
    button.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(code.innerText); button.textContent = 'Copied'; }
      catch {
        const range = document.createRange(); range.selectNodeContents(code);
        const sel = getSelection(); sel.removeAllRanges(); sel.addRange(range); button.textContent = 'Selected';
      }
      setTimeout(() => { button.textContent = 'Copy'; }, 1600);
    });
  });
})();
