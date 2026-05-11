import streamlit as st
import numpy as np
import cv2
import base64
import time

# --- CONFIG & PRESETS ---
LIGHT_PRESETS = {
    "Front": [0,0,1], "Top": [0,1,0.5], "Bottom": [0,-1,0.5], 
    "Left": [-1,0,0.5], "Right": [1,0,0.5], "Top-Left": [-0.7,0.7,0.5], 
    "Top-Right": [0.7,0.7,0.5], "Bottom-Left": [-0.7,-0.7,0.5], "Bottom-Right": [0.7,-0.7,0.5]
}

def guess_direction(filename):
    fn = filename.lower().replace(" ", "").replace("_", "").replace("-", "")
    mapping = {
        "topleft": "Top-Left", "topright": "Top-Right", "bottomleft": "Bottom-Left", "bottomright": "Bottom-Right",
        "top": "Top", "bottom": "Bottom", "left": "Left", "right": "Right", "front": "Front"
    }
    for key, val in mapping.items():
        if key in fn: return val
    return None

# --- CORE ENGINE ---

def solve_pbr_turbo(color_images, initial_lights, target_maps, prog_cb, iterations, flip_green, rough_mode, ao_mode):
    h, w, _ = color_images[0].shape
    results = {}
    
    prog_cb(0.1, "Analyzing Luminance...")
    gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for img in color_images]
    I_gray = np.array([img.flatten() for img in gray_imgs])
    L = np.array(initial_lights, dtype=np.float32)

    for i in range(iterations):
        pct = 0.1 + (i / iterations) * 0.5
        prog_cb(pct, f"Pass {i+1}/{iterations}")
        G_gray, _, _, _ = np.linalg.lstsq(L, I_gray, rcond=None)
        L = np.dot(I_gray, np.linalg.pinv(G_gray))
        L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-7)

    albedo_gray = np.linalg.norm(G_gray, axis=0)
    normals = G_gray / (albedo_gray + 1e-7)
    mean_n = np.mean(normals, axis=1, keepdims=True)
    normals[0:2, :] -= mean_n[0:2, :]
    normals /= (np.linalg.norm(normals, axis=0) + 1e-7)
    normal_map = normals.T.reshape(h, w, 3)

    prog_cb(0.75, "Baking PBR Channels...")
    if "Normal" in target_maps:
        if flip_green: normal_map[:,:,1] = -normal_map[:,:,1]
        results["Normal"] = ((normal_map + 1) / 2 * 255).astype(np.uint8)
    
    if "Albedo" in target_maps:
        color_albedo = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(3):
            I_c = np.array([img[:,:,i].flatten().astype(np.float32) / 255.0 for img in color_images])
            G_c, _, _, _ = np.linalg.lstsq(L, I_c, rcond=None)
            color_albedo[:,:,i] = np.linalg.norm(G_c, axis=0).reshape(h, w)
        results["Albedo"] = (np.clip(color_albedo[:,:,::-1], 0, 1) * 255).astype(np.uint8)

    if "Roughness" in target_maps:
        predicted_I = np.dot(L, G_gray); base = np.std(I_gray - predicted_I, axis=0).reshape(h, w)
        if rough_mode == "Standard": rough_raw = np.median(np.abs(I_gray - predicted_I), axis=0).reshape(h, w)
        elif rough_mode == "High-Pass": rough_raw = cv2.addWeighted(base, 1.5, cv2.GaussianBlur(base, (127, 127), 0), -0.5, 0)
        elif rough_mode == "Corrective (Flat-Field)": rough_raw = cv2.divide(base, cv2.GaussianBlur(base, (255, 255), 0) + 0.05, scale=1.0)
        results["Roughness"] = cv2.normalize(rough_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if "AO" in target_maps:
        if ao_mode == "Standard": ao_final = np.min(I_gray, axis=0).reshape(h, w)
        elif ao_mode == "Slope-Base (Geometric)":
            dx = cv2.Sobel(normal_map[:,:,0], cv2.CV_32F, 1, 0, ksize=3); dy = cv2.Sobel(normal_map[:,:,1], cv2.CV_32F, 0, 1, ksize=3)
            ao_final = 1.0 - cv2.normalize(np.sqrt(dx**2 + dy**2) * np.mean(np.abs(I_gray - np.dot(L, G_gray)), axis=0).reshape(h, w), None, 0, 1, cv2.NORM_MINMAX)
        results["AO"] = cv2.normalize(ao_final, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    prog_cb(1.0, "Complete!")
    return results

def three_js_viewer(maps_b64, active_view, exposure):
    html = f"""
    <!DOCTYPE html><html><head><style>body {{ margin:0; background:#111; overflow:hidden; }}</style></head>
    <body><script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const scene = new THREE.Scene();
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        
        // --- FILMIC TONE MAPPING ---
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = {exposure};
        renderer.outputEncoding = THREE.sRGBEncoding;
        
        document.body.appendChild(renderer.domElement);
        const loader = new THREE.TextureLoader();
        
        const mat = new THREE.MeshStandardMaterial({{ 
            map: {f"loader.load('data:image/jpeg;base64,{maps_b64['Albedo']}')" if active_view['Albedo'] else "null"},
            normalMap: {f"loader.load('data:image/jpeg;base64,{maps_b64['Normal']}')" if active_view['Normal'] else "null"},
            aoMap: {f"loader.load('data:image/jpeg;base64,{maps_b64['AO']}')" if active_view['AO'] else "null"},
            roughnessMap: {f"loader.load('data:image/jpeg;base64,{maps_b64['Roughness']}')" if active_view['Roughness'] else "null"},
            roughness: 1.0, metalness: 0.05
        }});
        
        if(mat.map) mat.map.encoding = THREE.sRGBEncoding;

        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(5, 5, 2, 2), mat);
        mesh.rotation.x = -Math.PI/2; scene.add(mesh);
        const cam = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 100);
        cam.position.set(0, 5, 5);
        
        // Balanced lighting for filmic roll-off
        const light = new THREE.PointLight(0xffffff, 3); scene.add(light);
        scene.add(new THREE.AmbientLight(0xffffff, 0.4));
        
        new THREE.OrbitControls(cam, renderer.domElement);
        function anim(t) {{ 
            requestAnimationFrame(anim); 
            light.position.set(Math.cos(t/1500)*4, 3, Math.sin(t/1500)*4); 
            renderer.render(scene, cam); 
        }}
        anim(0);
    </script></body></html>"""
    st.iframe(src=f"data:text/html;base64,{base64.b64encode(html.encode()).decode()}", height=550)

# --- UI APP ---
st.set_page_config(layout="wide", page_title="PBR Studio Pro")
st.markdown("""<style>.stDownloadButton { position: absolute; z-index: 10; top: 10px; left: 10px; }</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📥 Solver Settings")
    files = st.file_uploader("Upload Light Set", accept_multiple_files=True)
    st.divider()
    r_mode = st.selectbox("Roughness Logic", ["Standard", "High-Pass", "Corrective (Flat-Field)"], index=2)
    ao_mode = st.selectbox("AO Logic", ["Standard", "Slope-Base (Geometric)"], index=1)
    n_format = st.radio("Normal Standard", ["OpenGL (Y+)", "DirectX (Y-)"])
    iter_val = st.slider("Solver Iterations", 1, 20, 8)
    run_btn = st.button("🚀 SYNTHESIZE", use_container_width=True)
    
    if files:
        st.subheader("Light Directions")
        lights = []
        for i, f in enumerate(files):
            g = guess_direction(f.name); keys = list(LIGHT_PRESETS.keys())
            idx = keys.index(g) if g in keys else (i % len(keys))
            lights.append(LIGHT_PRESETS[st.selectbox(f.name, keys, index=idx, key=f"L_{i}")])

results_container = st.container()

if files and len(files) >= 3 and run_btn:
    imgs = [cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) for f in files]
    with results_container:
        st.subheader("⚙️ Synthesis in Progress")
        main_pb = st.progress(0)
        main_status = st.empty()
        def on_prog(val, text):
            main_pb.progress(val)
            main_status.code(f"SOLVER >> {text}")
        st.session_state.results = solve_pbr_turbo(imgs, lights, ["Albedo", "Normal", "Roughness", "AO"], on_prog, iter_val, ("DirectX" in n_format), r_mode, ao_mode)
        time.sleep(0.2)
        st.rerun()

if 'results' in st.session_state and st.session_state.results:
    res = st.session_state.results
    with results_container:
        v_col, t_col = st.columns([3, 1])
        with t_col:
            st.subheader("👁️ Viewport")
            v_albedo = st.toggle("Show Albedo", value=True)
            v_normal = st.toggle("Show Normals", value=True)
            v_rough = st.toggle("Show Roughness", value=True)
            v_ao = st.toggle("Show AO", value=True)
            st.divider()
            # New Exposure Control for Filmic Mapping
            v_exp = st.slider("Exposure", 0.1, 3.0, 1.0, step=0.1)
            view_state = {"Albedo": v_albedo, "Normal": v_normal, "Roughness": v_rough, "AO": v_ao}
        with v_col:
            st.subheader("🕹️ 3D Viewport")
            preview_b64 = {}
            for k, v in res.items():
                export_prep = cv2.cvtColor(v, cv2.COLOR_RGB2BGR) if len(v.shape) == 3 else v
                _, buffer = cv2.imencode('.jpg', export_prep, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                preview_b64[k] = base64.b64encode(buffer).decode()
            three_js_viewer(preview_b64, view_state, v_exp)
        st.divider()
        grid = st.columns(4)
        for i, name in enumerate(["Albedo", "Normal", "Roughness", "AO"]):
            if name in res:
                with grid[i]:
                    img = res[name]
                    _, b = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img)
                    st.download_button(label="💾", data=b.tobytes(), file_name=f"{name}.png", key=f"dl_{name}")
                    st.image(img, caption=name, use_container_width=True)
else:
    with results_container:
        st.info("👋 Upload images to begin synthesis.")