document.addEventListener('DOMContentLoaded', function() {
    // 创建观察器选项
    const options = {
        root: null,
        rootMargin: '0px',
        threshold: 0.2  // 当20%的内容可见时触发
    };

    // 创建观察器
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, options);

    // 观察所有section
    const sections = document.querySelectorAll('.hero, .product-showcase, .business-plan');
    sections.forEach(section => {
        observer.observe(section);
    });
}); 