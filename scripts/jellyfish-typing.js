document.addEventListener('DOMContentLoaded', function() {
    const text = "Our purifier offers customizable designs like jellyfish for mobility, with solar panels and AI monitoring providing real-time performance data directly to customers' devices.\n\nThis affordability, efficiency, and adaptability combination makes our purifier a reliable, sustainable choice, supporting healthier communities worldwide.";
    const typingElement = document.querySelector('.product-description');
    const cursorElement = document.createElement('span');
    let index = 0;

    // 添加光标元素
    cursorElement.className = 'typing-cursor';
    typingElement.appendChild(cursorElement);
    
    // 清空初始文本
    typingElement.textContent = '';
    typingElement.appendChild(cursorElement);

    function type() {
        if (index < text.length) {
            if (text.charAt(index) === '\n') {
                typingElement.insertBefore(
                    document.createElement('br'),
                    cursorElement
                );
            } else {
                typingElement.insertBefore(
                    document.createTextNode(text.charAt(index)),
                    cursorElement
                );
            }
            index++;
            setTimeout(type, 35);
        }
    }

    // 开始打字效果
    type();
}); 