// 获取模态框元素
const modal = document.getElementById('fullscreen-modal');
const modalImg = document.getElementById('fullscreen-image');

// 打开全屏
function openFullscreen(img) {
    modal.style.display = "block";
    modalImg.src = img.src;
}

// 关闭全屏
function closeFullscreen() {
    modal.style.display = "none";
}

// ESC键关闭全屏
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeFullscreen();
    }
});

// 阻止图片点击事件冒泡
modalImg.onclick = function(event) {
    event.stopPropagation();
}; 