import streamlit as st
import numpy as np
import cv2
import base64
from scipy.fftpack import fft2, ifft2

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

def integrate_normals_corrected(normals, h, w, high_pass_strength):
    nx, ny = normals[:,:,0], normals[:,:,1] 
    u, v = np.meshgrid(np.fft.fftfreq(w), np.fft.fftfreq(h))
    denom = u**2 + v**2
    denom[0, 0] = 1 
    H = (1j * u * fft2(nx) + 1j * v * fft2(ny)) / denom
    height = np.real(ifft2(H))
    if high_pass_strength > 0:
        blur_size = int(max(h, w) * (high_pass_strength / 100)) | 1
        height = height - cv2.GaussianBlur(height, (blur_size, blur_size), 0)
    return cv2.normalize(height, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# --- ENGINE WITH ADAPTIVE LABELS ---
def solve_pbr_pro_v8(color_images, initial_lights, target_maps, prog_cb, iterations, flip_green, r_mode, m_mode, h_strength, ao_mode):
    h, w, _ = color_images[0].shape
    results = {}
    
    prog_cb(0.02, "Syncing Image Luma Channels...")
    gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for img in color_images]
    I_gray = np.array([img.flatten() for img in gray_imgs])
    L = np.array(initial_lights, dtype=np.float32)

    # Photometric Stereo with Adaptive Feedback
    for i in range(iterations):
        solver_progress = 0.05 + (i / iterations) * 0.40
        
        # Adaptive UI Descriptions
        if i < 3:
            desc = "Initializing Surface Geometry..."
        elif i < 10:
            desc = "Refining Light-Shadow Intersections..."
        elif i < 20:
            desc = "Compensating for Specular Glints..."
        else:
            desc = "Polishing Surface Convergence..."
            
        prog_cb(solver_progress, f"Pass {i+1}/{iterations}: {desc}")
        
        G_gray, _, _, _ = np.linalg.lstsq(L, I_gray, rcond=None)
        L = np.dot(I_gray, np.linalg.pinv(G_gray))
        L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-7)

    prog_cb(0.45, "Finalizing Surface Normals...")
    albedo_gray = np.linalg.norm(G_gray, axis=0)
    normals_raw = G_gray / (albedo_gray + 1e-7)
    normal_map = normals_raw.T.reshape(h, w, 3)

    active_steps = [m for m in ["Normal", "Height", "Albedo", "Roughness", "Metallic", "AO"] if m in target_maps]
    base_progress = 0.45
    progress_per_map = (0.55) / max(1, len(active_steps))

    for idx, map_type in enumerate(active_steps):
        current_pct = base_progress + (idx * progress_per_map)
        prog_cb(current_pct, f"Baking {map_type} Map...")

        if map_type == "Normal":
            n_out = normal_map.copy()
            if flip_green: n_out[:,:,1] = -n_out[:,:,1]
            results["Normal"] = ((n_out + 1) / 2 * 255).astype(np.uint8)
        elif map_type == "Height":
            results["Height"] = integrate_normals_corrected(normal_map, h, w, h_strength)
        elif map_type == "Albedo":
            color_albedo = np.zeros((h, w, 3), dtype=np.float32)
            for i in range(3):
                I_c = np.array([img[:,:,i].flatten().astype(np.float32) / 255.0 for img in color_images])
                G_c, _, _, _ = np.linalg.lstsq(L, I_c, rcond=None)
                color_albedo[:,:,i] = np.linalg.norm(G_c, axis=0).reshape(h, w)
            results["Albedo"] = (np.clip(color_albedo[:,:,::-1], 0, 1) * 255).astype(np.uint8)

        predicted_I = np.dot(L, G_gray)
        residual = np.abs(I_gray - predicted_I)
        base_error = np.std(I_gray - predicted_I, axis=0).reshape(h, w)

        if map_type == "Roughness":
            if r_mode == "Standard": rough_raw = np.median(residual, axis=0).reshape(h, w)
            elif r_mode == "High-Pass": rough_raw = cv2.addWeighted(base_error, 1.5, cv2.GaussianBlur(base_error, (127, 127), 0), -0.5, 0)
            elif r_mode == "Corrective (Flat-Field)": rough_raw = cv2.divide(base_error, cv2.GaussianBlur(base_error, (255, 255), 0) + 0.05, scale=1.0)
            results["Roughness"] = cv2.normalize(rough_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        elif map_type == "Metallic":
            if m_mode == "None (Dielectric)":
                results["Metallic"] = None
            else:
                met_error = np.max(residual, axis=0).reshape(h, w)
                if m_mode == "Clean Specular (De-Ghost)":
                    clean_met = cv2.divide(met_error, cv2.GaussianBlur(met_error, (101, 101), 0) + 0.02)
                    results["Metallic"] = cv2.normalize(clean_met**2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                elif m_mode == "Specular Deviation": 
                    results["Metallic"] = cv2.normalize(met_error**2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                elif m_mode == "Luma Threshold": 
                    results["Metallic"] = cv2.threshold(cv2.normalize(met_error, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), 200, 255, cv2.THRESH_BINARY)[1]

        elif map_type == "AO":
            if ao_mode == "Standard": ao_final = np.min(I_gray, axis=0).reshape(h, w)
            elif ao_mode == "Slope-Base (Geometric)":
                dx = cv2.Sobel(normal_map[:,:,0], cv2.CV_32F, 1, 0, ksize=3); dy = cv2.Sobel(normal_map[:,:,1], cv2.CV_32F, 0, 1, ksize=3)
                ao_final = 1.0 - cv2.normalize(np.sqrt(dx**2 + dy**2) * base_error, None, 0, 1, cv2.NORM_MINMAX)
            results["AO"] = (ao_final * 255).astype(np.uint8)

    prog_cb(1.0, "Ready!")
    return results

# --- VIEWER & UI ---
def three_js_viewer(maps_b64, active_view, exposure, disp_scale, is_metallic, invert_height):
    metal_val = 1.0 if is_metallic else 0.0
    final_disp = -disp_scale if invert_height else disp_scale
    html = f"""
    <!DOCTYPE html><html><head><style>body {{ margin:0; background:#111; overflow:hidden; }}</style></head>
    <body><script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const scene = new THREE.Scene();
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = {exposure};
        document.body.appendChild(renderer.domElement);
        const loader = new THREE.TextureLoader();
        const mat = new THREE.MeshStandardMaterial({{ 
            map: {f"loader.load('data:image/jpeg;base64,{maps_b64['Albedo']}')" if active_view.get('Albedo') else "null"},
            normalMap: {f"loader.load('data:image/jpeg;base64,{maps_b64['Normal']}')" if active_view.get('Normal') else "null"},
            aoMap: {f"loader.load('data:image/jpeg;base64,{maps_b64['AO']}')" if active_view.get('AO') else "null"},
            roughnessMap: {f"loader.load('data:image/jpeg;base64,{maps_b64['Roughness']}')" if active_view.get('Roughness') else "null"},
            metalnessMap: {f"loader.load('data:image/jpeg;base64,{maps_b64['Metallic']}')" if (active_view.get('Metallic') and is_metallic) else "null"},
            displacementMap: {f"loader.load('data:image/jpeg;base64,{maps_b64['Height']}')" if active_view.get('Height') else "null"},
            displacementScale: {final_disp}, roughness: 1.0, metalness: {metal_val} 
        }});
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(5, 5, 256, 256), mat);
        mesh.rotation.x = -Math.PI/2; scene.add(mesh);
        const cam = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 100);
        cam.position.set(0, 5, 5);
        const light = new THREE.PointLight(0xffffff, 2.5); scene.add(light);
        scene.add(new THREE.AmbientLight(0xffffff, 0.4));
        new THREE.OrbitControls(cam, renderer.domElement);
        function anim(t) {{ requestAnimationFrame(anim); light.position.set(Math.cos(t/1500)*4, 3, Math.sin(t/1500)*4); renderer.render(scene, cam); }}
        anim(0);
    </script></body></html>"""
    st.iframe(src=f"data:text/html;base64,{base64.b64encode(html.encode()).decode()}", height=550)

st.set_page_config(layout="wide", page_title="PBR Studio Pro")

with st.sidebar:
    st.title("📥 Solver Config")
    files = st.file_uploader("Upload Light Set", accept_multiple_files=True)
    if files:
        with st.expander("📍 Assign Light Directions", expanded=True):
            lights = []
            for i, f in enumerate(files):
                g = guess_direction(f.name); keys = list(LIGHT_PRESETS.keys())
                idx = keys.index(g) if g in keys else (i % len(keys))
                lights.append(LIGHT_PRESETS[st.selectbox(f.name, keys, index=idx, key=f"L_{i}")])
    st.divider()
    with st.expander("🛠️ Map Selection", expanded=True):
        maps_to_bake = st.multiselect("Bake Targets", ["Albedo", "Normal", "Roughness", "Metallic", "Height", "AO"], default=["Albedo", "Normal", "Roughness", "Height"])
    with st.expander("🧪 Algorithms", expanded=False):
        r_mode = st.selectbox("Roughness", ["Standard", "High-Pass", "Corrective (Flat-Field)"], index=2)
        m_mode = st.selectbox("Metallic", ["Clean Specular (De-Ghost)", "Specular Deviation", "Luma Threshold", "None (Dielectric)"])
        ao_mode = st.selectbox("AO", ["Standard", "Slope-Base (Geometric)"], index=1)
        h_strength = st.slider("Height High-Pass %", 0, 50, 10)
        n_format = st.radio("Normal Standard", ["OpenGL", "DirectX"])
        iter_val = st.slider("Iterations", 1, 50, 20)
    run_btn = st.button("🚀 SYNTHESIZE", use_container_width=True)

results_container = st.container()

if files and len(files) >= 3 and run_btn:
    imgs = [cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) for f in files]
    with results_container:
        pb = st.progress(0)
        st.session_state.results = solve_pbr_pro_v8(imgs, lights, maps_to_bake, lambda v, t: pb.progress(v, text=t), iter_val, ("DirectX" in n_format), r_mode, m_mode, h_strength, ao_mode)
        st.session_state.is_metallic = (m_mode != "None (Dielectric)" and "Metallic" in maps_to_bake)
        st.rerun()

if 'results' in st.session_state:
    res = st.session_state.results
    v_col, t_col = st.columns([3, 1])
    with t_col:
        st.subheader("👁️ Viewport")
        view_state = {m: st.toggle(f"Show {m}", value=True) for m in res.keys() if res[m] is not None}
        st.divider()
        v_exp = st.slider("Exposure", 0.1, 3.0, 1.0)
        v_disp = st.slider("Height Scale", 0.0, 1.0, 0.15)
        v_inv = st.toggle("Invert Height Preview", value=False)
    with v_col:
        preview_b64 = {}
        for k, v in res.items():
            if v is not None:
                _, b = cv2.imencode('.jpg', cv2.cvtColor(v, cv2.COLOR_RGB2BGR) if len(v.shape)==3 else v, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                preview_b64[k] = base64.b64encode(b).decode()
        three_js_viewer(preview_b64, view_state, v_exp, v_disp, st.session_state.get('is_metallic', False), v_inv)
    st.divider()
    valid_res = {k: v for k, v in res.items() if v is not None}
    grid = st.columns(min(4, len(valid_res)))
    for i, (name, img) in enumerate(valid_res.items()):
        with grid[i % 4]:
            st.image(img, caption=name, use_container_width=True)
            _, b = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape)==3 else img)
            st.download_button("💾", b.tobytes(), f"{name}.png", key=f"dl_{name}")