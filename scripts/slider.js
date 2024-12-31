document.addEventListener('DOMContentLoaded', function() {
    const slides = document.querySelectorAll('.product-slide');
    const prevButton = document.querySelector('.prev-slide');
    const nextButton = document.querySelector('.next-slide');
    let currentSlide = 0;
    let isAnimating = false;

    function showSlide(index, direction = 'next') {
        if (isAnimating) return;
        isAnimating = true;

        const currentActive = document.querySelector('.product-slide.active');
        currentActive.style.transform = direction === 'next' ? 'translateX(-100%)' : 'translateX(100%)';
        currentActive.style.opacity = '0';
        currentActive.classList.remove('active');

        slides[index].style.transform = direction === 'next' ? 'translateX(100%)' : 'translateX(-100%)';
        slides[index].style.opacity = '0';
        
        // 强制重排
        slides[index].offsetHeight;

        slides[index].style.transform = 'translateX(0)';
        slides[index].style.opacity = '1';
        slides[index].classList.add('active');

        setTimeout(() => {
            isAnimating = false;
        }, 500);
    }

    function nextSlide() {
        currentSlide = (currentSlide + 1) % slides.length;
        showSlide(currentSlide, 'next');
    }

    function prevSlide() {
        currentSlide = (currentSlide - 1 + slides.length) % slides.length;
        showSlide(currentSlide, 'prev');
    }

    // 按钮事件监听
    nextButton.addEventListener('click', nextSlide);
    prevButton.addEventListener('click', prevSlide);

    // 触摸滑动支持
    let touchStartX = 0;
    let touchEndX = 0;

    const sliderContainer = document.querySelector('.slider-container');
    
    sliderContainer.addEventListener('touchstart', e => {
        touchStartX = e.changedTouches[0].screenX;
    }, { passive: true });

    sliderContainer.addEventListener('touchend', e => {
        touchEndX = e.changedTouches[0].screenX;
        const diff = touchStartX - touchEndX;
        
        if (Math.abs(diff) > 50) {  // 确保滑动距离足够
            if (diff > 0) {
                nextSlide();  // 向左滑动
            } else {
                prevSlide();  // 向右滑动
            }
        }
    }, { passive: true });
});

