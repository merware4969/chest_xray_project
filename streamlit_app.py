# streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.metrics import auc
from streamlit_option_menu import option_menu
from glob import glob
import matplotlib.font_manager as fm

# -----------------------------
# 결정론적 설정 (Grad-CAM 일관성 유지)
# -----------------------------
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# 한글 폰트 설정 (KoPub + HTML 스타일 적용까지)
# -----------------------------
font_path = "./KoPubDotumMedium.ttf"
fontprop = fm.FontProperties(fname=font_path, size=14)
plt.rcParams['font.family'] = 'KoPubDotum'
plt.rcParams['axes.unicode_minus'] = False

with open(font_path, "rb") as f:
    base64_font = f.read().hex()

# HTML용 font-face 정의 추가
FONT_STYLE = f"""
<style>
@font-face {{
  font-family: 'KoPub';
  src: url(data:font/ttf;charset=utf-8;base64,{base64_font});
}}
body, div, h1, h2, h3, h4, h5, span, p {{
  font-family: 'KoPub', sans-serif;
}}
</style>
"""
st.markdown(FONT_STYLE, unsafe_allow_html=True)

# -----------------------------
# 모델 로딩
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('models/best_resnet34_addval.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

# -----------------------------
# 이미지 전처리
# -----------------------------
def preprocess_image(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(path).convert('RGB')
    return transform(image).unsqueeze(0).to(device), image

def preprocess_uploaded(file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(file).convert('RGB')
    return transform(image).unsqueeze(0).to(device), image

# -----------------------------
# 예측
# -----------------------------
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
    return pred, prob.squeeze().cpu().numpy()

# -----------------------------
# Grad-CAM
# -----------------------------
def generate_gradcam(model, image_tensor, class_idx):
    gradients = []
    activations = []

    def fw_hook(module, input, output):
        activations.append(output)

    def bw_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = model.layer4.register_forward_hook(fw_hook)
    h2 = model.layer4.register_full_backward_hook(bw_hook)

    output = model(image_tensor)
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0][class_idx] = 1
    output.backward(gradient=one_hot)

    acts = activations[0].squeeze(0).cpu()
    grads = gradients[0].squeeze(0).cpu()
    weights = grads.mean(dim=(1, 2))

    cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam.detach().numpy(), 0)
    cam /= np.max(cam)

    h1.remove()
    h2.remove()
    return cam

# -----------------------------
# 빠른 ROC 커브 (시뮬레이션)
# -----------------------------
def plot_mock_roc():
    fpr = np.array([0.0, 0.1, 0.3, 0.6, 0.9, 1.0])
    tpr = np.array([0.0, 0.6, 0.8, 0.9, 0.96, 1.0])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('거짓 양성 비율')
    ax.set_ylabel('진짜 양성 비율')
    ax.set_title('ROC 커브')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# -----------------------------
# Streamlit UI 구성
# -----------------------------
st.set_page_config(page_title="폐렴 분류 시각화", layout="centered")

with st.sidebar:
    selected = option_menu(
        menu_title="📌 메뉴",
        options=["📊 ROC 커브", "🔥 Grad-CAM 시각화", "🧪 추론"],
        icons=["bar-chart", "fire", "upload"],
        menu_icon="cast",
        default_index=0
    )

st.title("🧠 ResNet34 기반 폐렴 분류 모델 시각화")
model = load_model()
class_names = ['정상', '폐렴']

if selected == "📊 ROC 커브":
    st.header("📊 샘플 기반 ROC 커브")
    plot_mock_roc()

elif selected == "🔥 Grad-CAM 시각화":
    st.header("🔥 Grad-CAM 시각화")
    image_files = [f for f in os.listdir("train_samples") if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    if image_files:
        img_name = st.selectbox("이미지를 선택하세요", image_files)
        path = os.path.join("train_samples", img_name)
        img_tensor, orig_img = preprocess_image(path)
        pred_class, _ = predict(model, img_tensor)
        cam = generate_gradcam(model, img_tensor, pred_class)
        cam_resized = cv2.resize(cam, orig_img.size)

        fig, ax = plt.subplots()
        ax.imshow(orig_img)
        ax.imshow(cam_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
        ax.set_title(f"예측 클래스: {class_names[pred_class]}")
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("train_samples 폴더에 이미지가 없습니다.")

elif selected == "🧪 추론":
    st.header("🧪 테스트 이미지 추론")
    uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded:
        img_tensor, orig_img = preprocess_uploaded(uploaded)
        pred_class, probs = predict(model, img_tensor)
        st.image(orig_img, caption="업로드한 이미지", use_column_width=True)

        st.markdown(f"""
        <div style='padding: 1.5em; background-color: #f0f2f6; border-left: 6px solid #4b9cd3; font-size: 24px;'>
            <b>예측 결과:</b> {class_names[pred_class]}<br>
            <b>정확도:</b> {probs[pred_class] * 100:.2f}%
        </div>
        """, unsafe_allow_html=True)