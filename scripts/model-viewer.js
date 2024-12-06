let scene, camera, renderer, controls, model;

function init() {
    // 创建场景 - 3D空间的容器
    scene = new THREE.Scene();
    
    // 设置渐变背景色，从浅蓝到深蓝
    const topColor = new THREE.Color(0x72B6E4);    // 浅蓝色（海面）
    const middleColor = new THREE.Color(0x3C7FB4); // 中蓝色
    const bottomColor = new THREE.Color(0x1E4976); // 深蓝色（海底）
    
    const canvas = document.createElement('canvas');
    canvas.width = 2;
    canvas.height = 512;
    
    const context = canvas.getContext('2d');
    const gradient = context.createLinearGradient(0, 0, 0, 512);
    gradient.addColorStop(0, topColor.getStyle());
    gradient.addColorStop(0.5, middleColor.getStyle());
    gradient.addColorStop(1, bottomColor.getStyle());
    
    context.fillStyle = gradient;
    context.fillRect(0, 0, 2, 512);
    
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    
    scene.background = texture;
    
    // 添加轻微的雾效果增加深度感
    scene.fog = new THREE.FogExp2(0x3C7FB4, 0.02);

    // 创建透视相机
    camera = new THREE.PerspectiveCamera(
        40,      // fov (Field of View) - 视野角度，值越大视野范围越大
        document.getElementById('coral-model').clientWidth / document.getElementById('coral-model').clientHeight,  // aspect - 宽高比
        0.01,    // near - 近平面距离，小于这个距离的物体不会被渲染
        1000     // far - 远平面距离，大于这个距离的物体不会被渲染
    );
    // 设置相机位置 (x, y, z)
    // x: 左右位置，正值向右
    // y: 上下位置，正值向上
    // z: 前后位置，正值后（远离物体）
    camera.position.set(0, 2, 4);

    // 创建WebGL渲染器
    renderer = new THREE.WebGLRenderer({ 
        antialias: true,  // 启用抗锯齿
        alpha: true       // 启用透明背景
    });
    // 设置渲染器尺寸为容器尺寸
    renderer.setSize(
        document.getElementById('coral-model').clientWidth,
        document.getElementById('coral-model').clientHeight
    );
    renderer.setClearColor(0x000000, 0);  // 设置透明背景

    // 将渲染器的画布添加到DOM中
    document.getElementById('coral-model').appendChild(renderer.domElement);

    // 调整环境光 - 增加亮度
    const ambientLight = new THREE.AmbientLight(
        0x6699CC,  // 偏蓝色但更亮
        0.5        // 增加强度
    );
    scene.add(ambientLight);

    // 增强水面透光效果
    const surfaceLight = new THREE.DirectionalLight(0xFFFFFF, 0.6);  // 更亮的主光源
    surfaceLight.position.set(0, 5, 0);
    scene.add(surfaceLight);

    // 调整散射光
    const scatterLight1 = new THREE.SpotLight(0x88CCFF, 0.4);  // 增加亮度
    scatterLight1.position.set(2, 3, 2);
    scatterLight1.angle = Math.PI / 4;
    scene.add(scatterLight1);

    const scatterLight2 = new THREE.SpotLight(0x88CCFF, 0.4);
    scatterLight2.position.set(-2, 3, -2);
    scatterLight2.angle = Math.PI / 4;
    scene.add(scatterLight2);

    // 添加一个额外的补光
    const fillLight = new THREE.DirectionalLight(0xFFFFFF, 0.3);
    fillLight.position.set(0, 0, 5);  // 从前方打光
    scene.add(fillLight);

    // 添加轨道控制器 - 用于鼠标交互
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;           // 启用阻尼效果，使控制更平滑
    controls.dampingFactor = 0.1;           // 阻尼系数
    controls.screenSpacePanning = false;      // 禁用屏幕空间平移
    controls.minDistance = 2;                // 最小缩放距离
    controls.maxDistance = 15;               // 最大缩放距离

    // 添加坐标轴辅助工具
    const axesHelper = new THREE.AxesHelper(5);  // 参数5是轴的长度
    scene.add(axesHelper);

    // 加载3D型
    const loader = new THREE.GLTFLoader();
    loader.load(
        '/models/coral.gltf',
        function (gltf) {
            console.log('Model loaded successfully:', gltf);
            model = gltf.scene;
            
            // 调整模型变换
            model.scale.set(25, 25, 25);                    // 模型缩放
            model.position.set(0, -1, 0);                   // 模型位置
            model.rotation.set(Math.PI*1.5, Math.PI * 2, Math.PI *2 );  // x轴旋转180度（Math.PI = 180度）
            
            // 遍历模型中的有网格
            model.traverse((child) => {
                if (child.isMesh) {
                    child.material.needsUpdate = true;
                    child.material.metalness = 0.2;      // 略微增加金属感
                    child.material.roughness = 0.7;      // 降低粗糙度
                    child.material.envMapIntensity = 0.5; // 增加环境反射
                    child.material.transparent = true;
                    child.material.opacity = 0.95;
                }
            });
            
            scene.add(model);
            
            // 移除额外的光源，使用场景中已有的光源
            controls.update();
        },
        // 加载进度回调
        function (xhr) {
            if (xhr.lengthComputable) {
                const percentComplete = xhr.loaded / xhr.total * 100;
                console.log('Loading progress: ' + Math.round(percentComplete) + '%');
            }
        },
        // 错误回调
        function (error) {
            console.error('Error loading model:', error);
        }
    );

    // 监听窗口大小变化
    window.addEventListener('resize', onWindowResize, false);
}

// 动画循环函数
function animate() {
    requestAnimationFrame(animate);  // 求下一帧动画
    controls.update();              // 更新控制器
    renderer.render(scene, camera); // 渲染场景
}

// 窗口大小改变时更新渲染
function onWindowResize() {
    // 更新相机宽高比
    camera.aspect = document.getElementById('coral-model').clientWidth / document.getElementById('coral-model').clientHeight;
    camera.updateProjectionMatrix();
    // 更新渲染器尺寸
    renderer.setSize(
        document.getElementById('coral-model').clientWidth,
        document.getElementById('coral-model').clientHeight
    );
}

// 重置视角按钮功能 - 保持与初始设置相同的值
document.getElementById('reset-view').addEventListener('click', function() {
    camera.position.set(0, 2, 4);
    controls.target.set(0, 0, 0);    // 重置到原点
    controls.update();
});

// 初始化场景
init();
// 开始动画循环
animate(); 