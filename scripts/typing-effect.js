document.addEventListener('DOMContentLoaded', function() {
    const text = "Our innovation, AquaNova, is an all-natural water purifier combining sodium alginate's adsorption power with copper oxide's photocatalytic degradation.\n\nThis eco-friendly product is placed in water bodies to eliminate a wide range of organic pollutants, heavy metals, and pathogens, functioning without frequent replacements for low-cost, sustainable operation.\n\nOur products are designed for freshwater and marine ecosystems, which is perfect for applications such as lakes, household aquariums, fish farms, and rural areas in need of clean and safe drinking water. We also cater to industries, municipalities, and eco-conscious consumers.";
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