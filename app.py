from pathlib import Path
import json
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from model import build_model

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / 'models'
ASSET_DIR = ROOT / 'assets'
CONFIG = json.loads((MODEL_DIR / 'config.json').read_text(encoding='utf-8'))
CLASS_NAMES = json.loads((MODEL_DIR / 'class_names.json').read_text(encoding='utf-8'))
SIZE = int(CONFIG.get('input_size', [224, 224])[0])
MEAN, STD = CONFIG['normalization']['mean'], CONFIG['normalization']['std']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config(page_title='KrishiMitra | Rice Leaf Disease Classification', page_icon='🌾', layout='wide')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--forest:#103d2b;--green:#08763e;--ink:#183b2d;--muted:#63746a;--line:#e1ebe3}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}.stApp{background:#f7faf8}.block-container{max-width:1420px;padding-top:1rem;padding-bottom:3rem}
.brand{display:flex;align-items:center;gap:13px;padding:8px 0 18px;border-bottom:1px solid var(--line);margin-bottom:18px}.brand-icon{font-size:30px;background:#e8f5e9;border-radius:50%;padding:8px 12px}.brand-name{font:800 1.7rem Manrope;color:var(--forest)}.brand-sub{color:var(--muted);border-left:1px solid #cad8ce;padding-left:16px}
.hero{background:linear-gradient(110deg,#eef7f0,#e4f1e7);padding:34px;border-radius:20px;margin-bottom:20px}.hero h1{font:800 clamp(2rem,4vw,3.2rem) Manrope;color:var(--forest);line-height:1.1;margin:.4rem 0 1rem}.hero p{font-size:1.05rem;line-height:1.6;max-width:680px;color:#385446}.chip{display:inline-block;background:#d9eddf;color:#155c35;border-radius:20px;padding:7px 13px;font-weight:700;font-size:.85rem}
.panel{background:white;border:1px solid var(--line);border-radius:16px;padding:19px;margin-bottom:14px}.section-title{font:800 1.35rem Manrope;color:var(--forest);margin:1.1rem 0 .35rem}.muted{color:var(--muted);line-height:1.6}.result-title{font:800 1.8rem Manrope;color:var(--forest)}
[data-testid="stFileUploader"]{background:white;border:1px dashed #a9cbb4;border-radius:14px;padding:12px}.stButton>button{border-radius:10px;font-weight:700;min-height:2.8rem}.stButton>button[kind="primary"]{background:#08763e;border-color:#08763e} [data-testid="stMetric"]{background:white;border:1px solid var(--line);border-radius:12px;padding:12px}
footer{visibility:hidden}#MainMenu{visibility:hidden}
</style>''', unsafe_allow_html=True)

FRIENDLY = {
'bacterial_leaf_blight':('Bacterial leaf blight','A bacterial disease that commonly causes water-soaked lesions and yellowing or drying that may progress from leaf tips and margins.','Check whether yellowing or drying begins near the leaf tip and extends along the leaf. Look at several leaves and plants.'),
'bacterial_leaf_streak':('Bacterial leaf streak','A bacterial disease associated with narrow, elongated, water-soaked streaks that may later turn brown.','Look for thin, long streaks rather than separate oval spots. Examine leaves against the light.'),
'bacterial_panicle_blight':('Bacterial panicle blight','A disease mainly affecting panicles and grain development; a leaf-only photo may not show its key signs.','Inspect panicles and grains as well as leaves. A leaf image alone is not enough to confirm this class.'),
'blast':('Rice blast','A fungal disease that can produce spindle-shaped lesions, often with pale centres and darker brown margins.','Look for spindle-shaped spots with grey or pale centres and darker edges. Check multiple leaves.'),
'brown_spot':('Brown spot','A fungal disease that can cause oval or circular brown lesions, sometimes with yellow halos.','Look for distinct oval brown spots and compare their size and distribution across several leaves.'),
'dead_heart':('Dead heart','A symptom in which the central shoot of a young rice tiller dries and may pull out easily.','Inspect the central shoot and the whole tiller. A single leaf close-up may not show the relevant symptom.'),
'downy_mildew':('Downy mildew','A disease that may cause pale yellow or whitish streaks and abnormal leaf growth.','Check for pale streaking and distorted growth on several leaves. Confirm with a local agriculture expert.'),
'hispa':('Rice hispa damage','Feeding or scraping damage from the rice hispa beetle; leaves may show white streaks or scraped patches.','Look for scraped leaf surfaces and white streaks; inspect the plant for small beetles or larvae.'),
'normal':('No disease pattern detected','The model assigned the image to its normal-leaf category. This does not prove the plant is disease-free.','Continue monitoring the crop and inspect other leaves. A healthy-looking leaf does not rule out other problems.'),
'tungro':('Rice tungro','A viral disease complex that may cause yellow-orange leaf discoloration and stunted plant growth.','Check leaf colour, plant height and whether symptoms are spreading in the field.')}

TRANSFORM = transforms.Compose([transforms.Resize((SIZE,SIZE)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])

@st.cache_resource(show_spinner='Loading rice-leaf classifier…')
def load_model():
    model = build_model(len(CLASS_NAMES)).to(DEVICE)
    weights_path = MODEL_DIR / CONFIG['primary_prediction_model']
    if not weights_path.exists():
        raise FileNotFoundError(f'Model checkpoint not found: {weights_path}')
    state = torch.load(weights_path,map_location=DEVICE,weights_only=True)
    model.load_state_dict(state,strict=True)
    model.eval()
    return model

def gradcam(model, tensor, class_index):
    saved = {}
    target_layer = model.features[-1]
    def capture(_module, _inputs, output):
        saved['activation'] = output
        output.register_hook(lambda grad: saved.__setitem__('gradient', grad))
    handle = target_layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(tensor.clone().detach().requires_grad_(True))
            logits[0,class_index].backward()
        acts, grads = saved['activation'], saved.get('gradient')
        if grads is None: raise RuntimeError('Gradients were not captured for Grad-CAM.')
        weights = grads.mean(dim=(2,3),keepdim=True)
        heat = torch.relu((weights*acts).sum(dim=1,keepdim=True))
        heat = F.interpolate(heat,size=(SIZE,SIZE),mode='bilinear',align_corners=False)[0,0]
        heat = heat.detach().cpu().numpy()
        lo, hi = float(heat.min()), float(heat.max())
        if hi-lo < 1e-8: raise RuntimeError('Grad-CAM returned a nearly uniform map.')
        return np.clip((heat-lo)/(hi-lo),0,1)
    finally:
        handle.remove(); model.zero_grad(set_to_none=True); model.eval()

def gradcam_overlay(image, heat):
    base = np.asarray(image.resize((SIZE,SIZE)).convert('RGB'),dtype=np.float32)
    # Visible blue-to-cyan-to-yellow-to-red heatmap without extra packages.
    h = np.clip(heat,0,1)
    r = np.clip(1.5-np.abs(4*h-3),0,1)
    g = np.clip(1.5-np.abs(4*h-2),0,1)
    b = np.clip(1.5-np.abs(4*h-1),0,1)
    color = np.stack([r,g,b],axis=-1)*255
    return Image.fromarray(np.uint8(np.clip(.58*base+.42*color,0,255)))

def label(key): return FRIENDLY.get(key,(key.replace('_',' ').title(),'',''))[0]
def explain(key): return FRIENDLY.get(key,('', '', ''))[1:]
def page_heading(t,d):
    st.markdown(f'<div class="section-title">{t}</div><div class="muted">{d}</div>',unsafe_allow_html=True)

st.markdown('<div class="brand"><div class="brand-icon">🌾</div><div class="brand-name">KrishiMitra</div><div class="brand-sub">Rice Leaf Disease Classification</div></div>',unsafe_allow_html=True)
page = st.radio('Navigation',['Home','Detect Disease','Disease Guide','About'],key='page_nav',horizontal=True,label_visibility='collapsed')

if page == 'Home':
    hero_img = ASSET_DIR/'rice_field_hero.jpg'
    a,b=st.columns([1.1,.9],gap='large')
    with a:
        st.markdown('<div class="hero"><span class="chip">AI-assisted crop support</span><h1>Identify and Understand Rice Leaf Diseases</h1><p>Upload a clear rice-leaf photograph to view the model’s predicted class, compare other possible classes and see which image regions influenced the prediction.</p></div>',unsafe_allow_html=True)
        if st.button('Start Disease Detection →',type='primary',use_container_width=True): st.session_state['page_nav']='Detect Disease'; st.rerun()
    with b:
        if hero_img.exists(): st.image(Image.open(hero_img),use_container_width=True)
    st.markdown('<div class="section-title">What KrishiMitra provides</div>',unsafe_allow_html=True)
    cols=st.columns(3)
    for c,(head,desc) in zip(cols,[('10-class classification','Compares the uploaded image with the ten categories supported by this model.'),('Prediction explanation','Shows the leading class scores and explains what the scores mean.'),('Grad-CAM visualisation','Highlights image regions that influenced the selected class score.')]):
        with c: st.markdown(f'<div class="panel"><b>{head}</b><p class="muted">{desc}</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="muted">For best results, photograph one complete rice leaf in natural light, with the affected area in focus.</div>',unsafe_allow_html=True)

elif page == 'Detect Disease':
    page_heading('Analyse a rice-leaf image','Upload one clear image. The model returns its highest-scoring supported class; it cannot guarantee a field diagnosis.')
    up=st.file_uploader('Upload leaf image',type=['jpg','jpeg','png','webp'],help='Use a clear, unfiltered photograph of a rice leaf.')
    if up:
        import hashlib
        upload_token=hashlib.sha256(up.getvalue()).hexdigest()
        if st.session_state.get('km_upload_token') != upload_token:
            st.session_state.pop('km_result',None)
            st.session_state['km_upload_token']=upload_token
        try:
            image=ImageOps.exif_transpose(Image.open(up)).convert('RGB')
            if min(image.size)<120: st.error('Image is too small. Please upload a clearer photograph.')
            else:
                st.image(image,caption='Uploaded image',width=430)
                if st.button('Classify image',type='primary',use_container_width=True):
                    with st.spinner('Classifying image and generating Grad-CAM…'):
                        model=load_model(); x=TRANSFORM(image).unsqueeze(0).to(DEVICE)
                        with torch.no_grad(): probs=torch.softmax(model(x),dim=1)[0].cpu().numpy()
                        order=np.argsort(probs)[::-1]; idx=int(order[0]); cam=gradcam(model,x,idx)
                        st.session_state['km_result']={'image':image,'probs':probs,'order':order,'idx':idx,'cam':cam}
                result=st.session_state.get('km_result')
                if result is not None:
                    p=result['probs']; order=result['order']; idx=result['idx']; key=CLASS_NAMES[idx]
                    top_score=float(p[idx]); second_score=float(p[int(order[1])]); margin=top_score-second_score
                    st.markdown('---'); page_heading('Classification result','This is the model’s ranking for the uploaded image, not a confirmed diagnosis.')
                    r1,r2=st.columns([1.1,.9],gap='large')
                    with r1:
                        st.markdown(f'<div class="panel"><div class="muted">Highest-scoring class</div><div class="result-title">{label(key)}</div><p class="muted">{explain(key)[0]}</p></div>',unsafe_allow_html=True)
                        m1,m2=st.columns(2); m1.metric('Model score',f'{top_score:.1%}'); m2.metric('Lead over 2nd class',f'{margin:.1%}')
                        if margin < .10: st.warning('The top two classes are close. Treat this result as uncertain and upload another clear image or seek expert confirmation.')
                        elif top_score < .55: st.warning('The top class has a relatively low score. The model may not recognise this image reliably.')
                        else: st.info('The model favours this class over the other supported classes. A high score is not proof that the prediction is correct.')
                    with r2:
                        st.markdown('<div class="panel"><b>How to read this result</b><p class="muted">The model calculates a score for each of its ten learned categories. The category with the largest score is shown first. The lead is the difference between the first and second scores. A small lead means the model found the image difficult to distinguish.</p><p class="muted"><b>Important:</b> these are softmax scores, not medically or agriculturally calibrated probabilities.</p></div>',unsafe_allow_html=True)
                    page_heading('Other classes considered','The next highest scores show which categories the model also considered.')
                    for j in order[:5]:
                        j=int(j); st.write(f'**{label(CLASS_NAMES[j])}** — {p[j]:.1%}'); st.progress(float(p[j]))
                    page_heading('Visual explanation (Grad-CAM)','The heatmap shows regions that contributed to the selected class score. It does not identify or outline disease lesions.')
                    c1,c2=st.columns(2)
                    with c1: st.image(result['image'],caption='Original uploaded image',use_container_width=True)
                    with c2: st.image(gradcam_overlay(result['image'],result['cam']),caption='Grad-CAM heatmap overlay',use_container_width=True)
                    st.markdown('<div class="panel"><b>What to check in the field</b><p class="muted">'+explain(key)[1]+'</p><p class="muted">Compare the result with several leaves and the whole plant. For treatment decisions, confirm with an agriculture officer or Krishi Vigyan Kendra. Do not choose pesticides based only on this prediction.</p></div>',unsafe_allow_html=True)
                    with st.expander('Why might the prediction be wrong?'):
                        st.write('Similar-looking symptoms, poor lighting, blur, background objects, unfamiliar varieties, or differences between training images and field photographs can lead to misclassification. Grad-CAM is only an explanation aid and does not prove that the model focused on the disease itself.')
        except Exception as e:
            st.error('The image could not be classified. Please try a clear JPG, PNG or WEBP image.')
            with st.expander('Technical details'): st.code(str(e))
    else:
        st.markdown('<div class="panel"><b>Ready for image analysis</b><p class="muted">Upload a rice-leaf photograph to see the predicted category, alternative classes and Grad-CAM visual explanation.</p></div>',unsafe_allow_html=True)

elif page == 'Disease Guide':
    page_heading('Rice disease guide','General visual signs for the ten categories supported by the model. These descriptions are for learning, not diagnosis.')
    for i,key in enumerate(CLASS_NAMES):
        title,desc,tip=FRIENDLY[key]
        with st.expander(title):
            st.write(desc); st.markdown('**What to inspect:** '+tip)
            thumb=ASSET_DIR/f'{key}.jpg'
            if thumb.exists(): st.image(Image.open(thumb),width=300)

else:
    page_heading('About KrishiMitra','A research-based rice-leaf image classification prototype.')
    st.markdown('<div class="panel"><b>Model</b><p class="muted">The public prediction function uses a MobileNetV3-Small image classifier and the supplied ten-class checkpoint. Grad-CAM provides a visual explanation of the selected class score.</p><b>Limitations</b><p class="muted">This app is not a certified plant diagnostic service. It does not reliably reject every non-rice image, does not provide a validated treatment recommendation, and has not been established as a substitute for field inspection by an agricultural expert.</p></div>',unsafe_allow_html=True)

st.markdown('<hr><div class="muted" style="text-align:center;font-size:.82rem">KrishiMitra · Rice-leaf classification support · Confirm important crop decisions with an agricultural expert</div>',unsafe_allow_html=True)
