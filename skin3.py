import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from skimage import feature
import time
from deepface import DeepFace
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image
import os
import base64






def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# Streamlit settings and styles
st.set_page_config(page_title="Face Analysis", page_icon=":smiley:")

@st.cache_data
def load_image(img_path):
    return Image.open(img_path)

st.sidebar.image(load_image("myskin.png"), width=150)  # You can adjust the width as desired.

# Navbar interaction using horizontal radio buttons in the sidebar
action = st.sidebar.radio("--MENU--", ("Home", "Skin Analysis", "About Us"), key="navbar")  # Radio buttons for interaction in the sidebar

# Add an open left sidebar
st.sidebar.title("Our Motto")

# Write content in the left sidebar
st.sidebar.write("Myskin.ai is more than just a skincare app. It's your partner on your skin health journey. We're here to help you understand your skin, care for your skin, and achieve your skin goals.")
st.sidebar.write("STEP 1: Turn on your camera or upload image")
st.sidebar.write("STEP 2: Click analyse to get your report & share with friends (optional)")
st.sidebar.write("STEP 3: Discover brands that are developing products for your skin type.")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .big-font {
        font-size:50px !important;
        color: #5A9;
    }
    .hover:hover {
        background-color: #f5f5f5;
        border-radius: 10px;
    }
    div[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        background-color: #f0f2f6;
        border-radius: 15px 15px 0 0;
    }
    .block-container {
        display: flex;
        margin-left: auto;
        margin-right: auto;
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
    }
    .big-font {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Mediapipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# Load Model
MODEL_PATH = 'more_data(3).h5'
new_model = load_model(MODEL_PATH)

@st.cache_data
def compute_lbp_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 2
    n_points = 24
    lbp = feature.local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype('float')
    return np.sum(lbp_hist)

@st.cache_data
def draw_landmarks_with_flicker(image):
    results = face_mesh.process(image)
    landmarks_image = np.zeros_like(image, dtype=np.uint8)
    
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                start_idx = connection[0]
                end_idx = connection[1]

                start_point = (int(landmarks.landmark[start_idx].x * image.shape[1]),
                               int(landmarks.landmark[start_idx].y * image.shape[0]))
                end_point = (int(landmarks.landmark[end_idx].x * image.shape[1]),
                             int(landmarks.landmark[end_idx].y * image.shape[0]))

                cv2.line(landmarks_image, start_point, end_point, (220, 220, 220), 1, lineType=cv2.LINE_AA)
                
                # Draw the landmark points
                cv2.circle(landmarks_image, start_point, 1, (127, 127, 127), -1)

    # Now, apply a slight blur to make the lines appear thinner
    landmarks_image = cv2.GaussianBlur(landmarks_image, (3, 3), 0)
    
    # Blend the original image with the landmarks image for a translucent effect
    alpha = 0.35
    blended_image = cv2.addWeighted(image, 1 - alpha, landmarks_image, alpha, 0)
    
    return blended_image

@st.cache_data
def count_wrinkles_and_spots(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray_roi, 9, 80, 80)
    edges = cv2.Canny(bilateral, 50, 150)
    
    wrinkles = np.sum(edges > 0)
    
    # Use adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Use morphological operations to fill small holes and remove small noises
    kernel = np.ones((3,3), np.uint8)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours to reduce noise
    min_spot_area = 4
    spots = len([cnt for cnt in contours if cv2.contourArea(cnt) > min_spot_area])
    
    return wrinkles, spots

@st.cache_data
def count_features(image):
    wrinkles, spots = count_wrinkles_and_spots(image)
    texture = compute_lbp_texture(image)
    return wrinkles, spots, texture


def detect_skin_type(oil, texture, acne):
    """
    oil      -> oiliness score (0–100)
    texture  -> texture score (higher = rough/dry)
    acne     -> acne probability (0–100)
    """

    if oil > 65 and acne > 40:
        return "Oily Skin"

    elif oil < 35 and texture > 50:
        return "Dry Skin"

    elif 40 <= oil <= 60 and acne > 30:
        return "Combination Skin"

    else:
        return "Normal Skin"



def mark_acne_and_pigmentation(face_bgr):
    h, w = face_bgr.shape[:2]

    # Central face mask (ignore borders)
    mask = np.zeros((h, w), dtype=np.uint8)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        face_mask,
        (w // 2, h // 2),
        (w // 4, h // 2),
        0, 0, 360, 255, -1
    )
    
    marked = face_bgr.copy()

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)

    red_channel = cv2.bitwise_and(face_bgr[:, :, 2], face_bgr[:, :, 2], mask=mask)

    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.equalizeHist(l)
    face_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)



    # ---------- ACNE DETECTION (RED SPOTS) ----------
    
    red_channel = cv2.bitwise_and(
        face_bgr[:, :, 2],
        face_bgr[:, :, 2],
        mask=face_mask
    )


    acne_mask = cv2.threshold(red_channel, 160, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 30 < area < 200:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.circle(marked, (x + w//2, y + h//2), 6, (0, 0, 255), 2)

        if len(contours) < 5:
            st.caption("Low confidence acne detection")
           

        acne_count = len([cnt for cnt in contours if 30 < cv2.contourArea(cnt) < 200])

       
             


    # ---------- PIGMENTATION (DARK PATCHES) ----------
    dark_mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)[1]
    dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=face_mask)

    dark_mask = cv2.medianBlur(dark_mask, 5)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 800:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = gray[y:y+h, x:x+w]

            if np.var(roi) < 120:
                cv2.circle(marked, (x + w//2, y + h//2), 8, (42, 42, 165), 2)            

    return marked


def acne_probability(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, 8, 1, method="uniform")

    texture_score = np.mean(lbp)
    redness = np.mean(face_bgr[:, :, 2])

    acne = (0.6 * min(texture_score / 20, 1) +
            0.4 * min(redness / 255, 1)) * 100
    return round(acne, 2)


def oiliness_score(face_bgr):
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    shine = np.mean(v) - np.mean(s)
    oil = np.clip((shine / 255) * 100, 0, 100)
    return round(oil, 2)


def pigmentation_score(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)

    pigment = np.clip((variance / 5000) * 100, 0, 100)
    return round(pigment, 2)


def skin_health_score(acne, oil, pigment):
    oil_penalty = abs(oil - 50)

    score = 100 \
            - (0.4 * acne) \
            - (0.3 * pigment) \
            - (0.3 * oil_penalty)

    return int(np.clip(score, 0, 100))

# ---------- FACE ALIGNMENT FUNCTIONS ----------

def is_face_inside_circle(face_box, img_w, img_h):
    x, y, w, h = face_box

    face_cx = x + w // 2
    face_cy = y + h // 2

    circle_cx = img_w // 2
    circle_cy = img_h // 2
    radius = min(img_w, img_h) // 3

    dist = ((face_cx - circle_cx) ** 2 + (face_cy - circle_cy) ** 2) ** 0.5
    return dist < radius


def detect_and_crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, "❌ Face not detected"

    img_h, img_w = frame.shape[:2]
    x, y, w, h = faces[0]

    if not is_face_inside_circle((x, y, w, h), img_w, img_h):
        return None, "⚠️ Please align your face inside the circle"

    margin = int(0.2 * w)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_w, x + w + margin)
    y2 = min(img_h, y + h + margin)

    face_crop = frame[y1:y2, x1:x2]
    return face_crop, "✅ Face aligned perfectly"


def draw_alignment_circle(frame):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    radius = min(w, h) // 3
    cv2.circle(frame, center, radius, (0, 255, 0), 3)
    return frame

def create_face_guide_image(width=640, height=480):
    guide = np.ones((height, width, 3), dtype=np.uint8) * 240

    center = (width // 2, height // 2)
    radius = min(width, height) // 3

    cv2.circle(guide, center, radius, (0, 200, 0), 6)

    cv2.putText(
        guide,
        "Place your face inside the circle",
        (width // 2 - 170, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (50, 50, 50),
        2
    )

    return guide



@st.cache_data
def process_image(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    frame = np.array(image)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    wrinkles, spots, texture = count_features(frame)
    frame = draw_landmarks_with_flicker(frame)

    return frame, wrinkles, spots, texture

@st.cache_data
def loadImage(filepath):
    test_img = tf_image.load_img(filepath, target_size=(180, 180))
    test_img = tf_image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img /= 255
    return test_img

@st.cache_data
def model_predict(img_path):
    global new_model
    age_pred = new_model.predict(loadImage(img_path))
    x = age_pred[0][0]
    rounded_age_value = round(x)  # Rounds 24.56 to 25
    age = 'About '+ str(rounded_age_value) +' years old'
    return age

# Streamlit UI
if action == "Home":
    # Home Page Cont.
    # .ent
    st.markdown("<div class='big-font'>MySkin.ai</div>", unsafe_allow_html=True)
    
    # Creating two divs side-by-side using HTML and CSS within Markdown
    st.markdown("""
    <div style="display: flex;">
        <div style="flex: 50%; padding: 10px; box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1); margin-right: 10px;">
            <h2>Why it matters?</h2>
            Taking care of your skin is important for both your physical and mental health. Healthy skin can help to protect you from the elements, reduce your risk of skin cancer, and boost your self-confidence. With so much information out there about skincare, it can be difficult to know where to start. Myskin.ai takes the guesswork out of skincare by providing you with personalised insights and recommendations based on your unique skin needs.
        </div>
        <div style="flex: 50%; padding: 10px; box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);">
            <h2>How it works?</h2>
            Turn on your camera or upload a picture to use our AI to analyse your skin and provide you with a report on your skin health, including:
            <ul>
                <li>Your skin type</li>
                <li>Your skin tone</li>
                <li>Your skin hydration levels</li>
                <li>Your skin elasticity</li>
                <li>The presence of any skin conditions, such as acne, rosacea, or eczema</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif action == "Skin Analysis":   
    

    st.markdown("<div class='big-font'>Face Analysis App</div>", unsafe_allow_html=True)

    # STEP 1: SELECT IMAGE SOURCE
    source = st.radio(
        "Choose Image Source",
        ["📁 Upload Image", "📷 Capture from Camera"],
        horizontal=True
        
    )

    image_input = None
    temp_image_path = None

    # STEP 2: INPUT
    if source == "📁 Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a clear face image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image_input = Image.open(uploaded_file).convert("RGB")
            temp_image_path = "temp_upload.jpg"
            image_input.save(temp_image_path)

    elif source == "📷 Capture from Camera":
        
         with st.container():

            overlay_base64 = image_to_base64("assets/face_overlay.png")

            st.markdown(f"""
            <style>
            .camera-wrapper {{             
                position: absolute;
                width: 380px;
                margin: auto;
                height: 380px;
                left: 0;
                right: 0;
                top: 43px;
            }}

            .face-overlay {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 10;
                pointer-events: none;
            }}

            .camera-wrapper iframe {{
                width: 100% !important;
                height: 100% !important;
            }}
            .stCameraInput > div {{ min-height:420px; background:#f1f2f6;}}
            </style>

            <div class="camera-wrapper">
                <img src="data:image/png;base64,{overlay_base64}" class="face-overlay">
            </div>
            """, unsafe_allow_html=True)

         

            # CAMERA INPUT

            camera_image = st.camera_input(
                "Align your face inside the circle",
                key="camera"
            )


            if camera_image:
                image_input = Image.open(camera_image).convert("RGB")
                frame = np.array(image_input)

                face_crop, status = detect_and_crop_face(frame)

                if face_crop is None:
                    st.error(status)
                    st.stop()

                st.success(status)

                image_input = Image.fromarray(face_crop)
                temp_image_path = "temp_face.jpg"
                image_input.save(temp_image_path)

            


    # STEP 3: PROCESS (COMMON)
    if image_input is not None:

        st.image(image_input, caption="Input Image", use_column_width=True)

        frame = np.array(image_input)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if np.mean(frame) < 60:
            st.warning("⚠️ Image is too dark. Please use better lighting.")


        with st.spinner("Analyzing skin..."):

            wrinkles, spots, texture = count_features(frame)
            frame_landmarks = draw_landmarks_with_flicker(frame)

            face_bgr = cv2.cvtColor(frame_landmarks, cv2.COLOR_RGB2BGR)

            acne = acne_probability(face_bgr)
            oil = oiliness_score(face_bgr)
            pigment = pigmentation_score(face_bgr)
            health = skin_health_score(acne, oil, pigment)
            skin_type = detect_skin_type(oil, texture, acne)

            marked = mark_acne_and_pigmentation(face_bgr)
            marked_rgb = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)

        st.image(marked_rgb, caption="Acne & Pigmentation Highlighted", use_column_width=True)

        st.subheader("🧴 Skin Analysis Results")
        st.metric("Acne Probability", f"{acne}%")
        st.metric("Oiliness", f"{oil}%")
        st.metric("Pigmentation", f"{pigment}%")
        st.metric("Skin Health", f"{health}/100")
        st.markdown(f"### 🧬 Skin Type: **{skin_type}**")

        if temp_image_path:
            age = model_predict(temp_image_path)
            st.subheader("🎂 Skin Age Analysis")
            st.write(age)

        def min_max_scale(value, min_value, max_value):
            """Scales a given value between 0 and 100 using Min-Max scaling."""
            return (value - min_value) / (max_value - min_value) * 100
    
        # Define some hypothetical maximum values for wrinkles, spots, and texture 
        # based on your dataset or domain knowledge.
        MAX_WRINKLES = 100000  # Just a placeholder value; adjust accordingly
        MAX_SPOTS = 100000     # Just a placeholder value; adjust accordingly
        MAX_TEXTURE = 100000   # Just a placeholder value; adjust accordingly
        
        # Use the min_max_scale function to scale each feature value to [0, 100]
        scaled_wrinkles = min_max_scale(wrinkles, 0, MAX_WRINKLES)
        scaled_spots = min_max_scale(spots, 0, MAX_SPOTS)
        scaled_texture = min_max_scale(texture, 0, MAX_TEXTURE)
        
        # Display the scaled scores in Streamlit
        
        st.markdown(f"**Standardized Wrinkles Score:** {scaled_wrinkles:.2f}")
        st.markdown(f"**Standardized Spots Score:** {scaled_spots:.2f}")
        st.markdown(f"**Standardized Texture Score:** {scaled_texture:.2f}")
    
        st.markdown("</div>", unsafe_allow_html=True)


        st.info("This analysis is cosmetic and not a medical diagnosis.")

        import os

        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

 
     

elif action == "About Us":
    # Custom CSS to set font size for a specific class
    st.markdown("""
    <style>
        .quote {
            font-size:24px !important;
            font-style: italic;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    # About Page Content
    st.markdown("<div class='big-font'>About Us...</div>", unsafe_allow_html=True)
    # Display the quote in italics using markdown
    st.markdown('<div class="quote">"We use digital AI tool to assess your skin health and provide you with personalised insights and recommendations."</div>', unsafe_allow_html=True)

st.write("---")
st.write("MySkin.ai provides AI-powered cosmetic skin insights, not medical diagnosis — helping users understand, track, and improve their skin over time.")

# Footer styles and content
footer_style = """
<style>
.footer {
  position: fixed;
  bottom: 0;
  left: 0;
  width: 100%;
  background-color: #f1f1f1;
  text-align: center;
  padding: 10px;
}
.footer-text p{
    font-size: 14px;
    font-style: italic;
}
.footer-logo {
        width: 80px; 
        height: auto;
        margin-right: 10px;
    }
</style>
"""
st.markdown(footer_style, unsafe_allow_html=True)

@st.cache_data
def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
logo_base64 = image_to_base64("Hexis-Lab-Logo.png")
footer_content = f"""
<div class="footer">
     <span class="footer-text"><p>ColorPixel.in</p></span>
</div>
"""
st.markdown(footer_content, unsafe_allow_html=True) 
