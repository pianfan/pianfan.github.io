let scene, camera, renderer, controls, model;

function init() {
    // 创建场景
%    scene.background = new THREE.Color(0xf0f8ff);

    // 创建相机
    camera = new THREE.PerspectiveCamera(
        45,
        document.getElementById('coral-model').clientWidth / document.getElementById('coral-model').clientHeight,
        0.01,
        1000
    );
    camera.position.set(0, 2, 1.5);

    // 创建渲染器
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    document.getElementById('coral-model').appendChild(renderer.domElement);

    // 添加环境光和方向光
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 2, 2);
    scene.add(directionalLight);

    // 添加轨道控制
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 3;
    controls.maxDistance = 8;

    // 修改加载部分的代码
    const loader = new THREE.GLTFLoader();

    // 配置 DRACO 解码器（如果模型使用了 DRACO 压缩）
    const dracoLoader = new THREE.DRACOLoader();
    dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.4.1/');
    loader.setDRACOLoader(dracoLoader);

    console.log('Starting model loader...');
    console.log('Attempting to load model from:', './models/coral.gltf');

    loader.load(
        '/models/coral.gltf',
        function (gltf) {
            console.log('Model loaded successfully:', gltf);
            model = gltf.scene;
            
            // 调整模型位置和旋转
            model.scale.set(25, 25, 25);
            model.position.set(0, 0, 0);
            model.rotation.set(Math.PI * 0.5, Math.PI * 1, Math.PI * 2);
            
            model.traverse((child) => {
                if (child.isMesh) {
                    console.log('Found mesh:', child.name);
                    child.material.needsUpdate = true;
                    child.material.metalness = 0;
                    child.material.roughness = 1;
                    child.material.side = THREE.DoubleSide;
                }
            });
            
            scene.add(model);
            
            // 减弱光照强度
            // 环境光
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
            scene.add(ambientLight);
            
            // 主光源
            const mainLight = new THREE.PointLight(0xffffff, 1);
            mainLight.position.set(2, 3, 2);
            scene.add(mainLight);
            
            // 补光
            const fillLight = new THREE.PointLight(0xffffff, 0.7);
            fillLight.position.set(-1, 1, 2);
            scene.add(fillLight);
            
            // 调整控制器
            controls.target.set(0, 0, 0);
            controls.minDistance = 2;
            controls.maxDistance = 15;
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            controls.update();
        },
        function (xhr) {
            if (xhr.lengthComputable) {
                const percentComplete = xhr.loaded / xhr.total * 100;
                console.log('Loading progress: ' + Math.round(percentComplete) + '%');
            }
        },
        function (error) {
            console.error('Error loading model:', error);
        }
    );

    // 添加窗口大小调整监听
    window.addEventListener('resize', onWindowResize, false);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// 重置视角按钮功能
document.getElementById('reset-view').addEventListener('click', function() {
    camera.position.set(0, 2, 1.5);
    controls.target.set(0, 0, 0);
    controls.update();
});

// 初始化
init();
animate(); 