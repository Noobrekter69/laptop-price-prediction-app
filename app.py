import joblib
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## page config
st.set_page_config(page_title="Laptop Price Estimator", page_icon="💻", layout="wide")

## shared matplotlib style for the dark theme
ACCENT = "#f5a623"
TEXT = "#e8eaed"
GRID = "#2a2f3a"

def style_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("none")
    if title:  ax.set_title(title, color=TEXT, weight="bold", fontsize=12, pad=14)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT, labelpad=8)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT, labelpad=8)
    ax.tick_params(colors=TEXT)
    ax.grid(alpha=0.15, color=GRID)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color("#555")

## CSS: spacing, readable text, styled components
st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(1200px 600px at 20% -10%, #1b2333 0%, #0e1117 55%); }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp li,
    .stMarkdown, h1, h2, h3, h4 { color: #e8eaed !important; }
    h1 { font-weight: 800 !important; letter-spacing:-0.5px; }

    /* More generous padding around the main content */
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem;
                       max-width: 1250px; }

    section[data-testid="stSidebar"] { background: #12161f; border-right: 1px solid #232a36; }
    /* Space out sidebar widgets a little */
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stSlider { margin-bottom: 0.4rem; }

    .stButton > button {
        background: linear-gradient(90deg,#f5a623,#f7b955);
        color:#111111 !important;
        border:none; border-radius:10px; padding:0.65rem 1rem; font-weight:800;
        transition: filter .2s ease, transform .05s ease;
    }
    /* Force the label text inside the button to stay black too */
    .stButton > button p,
    .stButton > button span,
    .stButton > button div { color:#111111 !important; }
    .stButton > button:hover { filter:brightness(1.05); }
    .stButton > button:active { transform: scale(0.99); }

    /* KPI cards with more internal padding + breathing room */
    .kpi { background:#161b26; border:1px solid #232a36; border-radius:14px;
           padding:22px 20px; text-align:center; margin-bottom:16px; }
    .kpi .k-label { font-size:12px; letter-spacing:1.5px; color:#9aa3b2 !important; }
    .kpi .k-value { font-size:26px; font-weight:800; color:#f5a623 !important; margin-top:8px; }

    /* Tab labels a touch bigger with spacing */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

## loading model
@st.cache_resource
def load_model():
    return joblib.load("laptop_price_model.pkl")

try:
    model = load_model()
    feature_names = list(model.feature_names_in_)
except FileNotFoundError:
    st.error("Model file 'laptop_price_model.pkl' was not found. Place it next to this app.")
    st.stop()
except Exception as e:
    st.error(f"The model could not be loaded: {e}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []


def build_row(ram, weight, ppi, cpu_ghz, ssd, hdd, inches,
              touchscreen, ips, company, type_name, cpu_tier, gpu_brand, os_choice):
    row = {f: 0 for f in feature_names}
    row['Ram_GB'] = ram; row['Weight_kg'] = weight; row['PPI'] = ppi
    row['Cpu_GHz'] = cpu_ghz; row['SSD_GB'] = ssd; row['HDD_GB'] = hdd
    row['Inches'] = inches
    row['Touchscreen'] = 1 if touchscreen else 0
    row['IPS'] = 1 if ips else 0
    for col in [f"Company_{company}", f"TypeName_{type_name}",
                f"Cpu_tier_{cpu_tier}", f"Gpu_brand_{gpu_brand}", f"OS_{os_choice}"]:
        if col in row:
            row[col] = 1
    return row


def predict_for(**kw):
    return float(model.predict(pd.DataFrame([build_row(**kw)])[feature_names])[0])


def laptop_animation_html(price_text: str) -> str:
    return f"""
    <div style="display:flex;justify-content:center;align-items:flex-end;height:260px;
                font-family:-apple-system,Segoe UI,Roboto,sans-serif;">
      <div class="laptop"><div class="lid"><div class="screen"><div class="sc">
        <div class="lab">ESTIMATED PRICE</div><div class="pr">{price_text}</div>
      </div></div></div><div class="base"></div></div>
    </div>
    <style>
      .laptop{{position:relative;width:300px;}}
      .lid{{width:300px;height:186px;transform-origin:bottom center;
        transform:perspective(1200px) rotateX(-90deg);
        animation:openLid 1s ease-out .2s forwards;}}
      .screen{{width:100%;height:100%;background:#0b0b0f;border:6px solid #1c1c22;
        border-radius:12px 12px 4px 4px;display:flex;align-items:center;
        justify-content:center;overflow:hidden;}}
      .sc{{text-align:center;color:#fff;opacity:0;animation:on .6s ease-out 1.2s forwards;}}
      .lab{{font-size:12px;letter-spacing:2px;color:#8ab4ff;margin-bottom:6px;}}
      .pr{{font-size:34px;font-weight:700;color:#fff;}}
      .base{{width:340px;height:15px;margin-left:-20px;
        background:linear-gradient(#d4d7dd,#b7bcc6);border-radius:4px 4px 10px 10px;
        box-shadow:0 8px 18px rgba(0,0,0,.25);}}
      @keyframes openLid{{from{{transform:perspective(1200px) rotateX(-90deg);}}
        to{{transform:perspective(1200px) rotateX(0deg);}}}}
      @keyframes on{{from{{opacity:0;}}to{{opacity:1;text-shadow:0 0 12px rgba(138,180,255,.5);}}}}
    </style>
    """


## side bar
with st.sidebar:
    st.header("Laptop specifications")
    company = st.selectbox("Brand", ['Dell','HP','Lenovo','Asus','Acer','MSI','Toshiba','Razer','Apple','Other'])
    type_name = st.selectbox("Laptop type", ['Notebook','Gaming','Ultrabook','Workstation','2 in 1 Convertible','Netbook'])
    inches = st.slider("Screen size (inches)", 10.0, 18.0, 15.6, step=0.1)
    weight = st.slider("Weight (kg)", 0.7, 4.5, 1.8, step=0.1)
    ppi = st.slider("Screen sharpness (PPI)", 90, 350, 141)
    touchscreen = st.checkbox("Touchscreen")
    ips = st.checkbox("IPS display")

    st.markdown("---")
    st.markdown("**Performance**")
    cpu_tier = st.selectbox("CPU tier", ['Core i3','Core i5','Core i7','Other'])
    cpu_ghz = st.slider("CPU speed (GHz)", 0.9, 3.6, 2.5, step=0.1)
    gpu_brand = st.selectbox("GPU brand", ['Intel','Nvidia','AMD'])
    os_choice = st.selectbox("Operating system", ['Windows','Mac','No OS'])
    ram = st.selectbox("RAM (GB)", [2,4,8,12,16,24,32,64], index=2)

    st.markdown("---")
    st.markdown("**Storage (GB)**")
    ssd = st.selectbox("SSD storage (GB)", [0,128,256,512,1024], index=2)
    hdd = st.selectbox("HDD storage (GB)", [0,500,1024,2048])

    st.markdown("")
    predict_clicked = st.button("Estimate price", type="primary", use_container_width=True)


## main
st.title("Laptop Price Estimator")
st.caption("Specification-based price estimates in SGD, for online sellers and small retailers.")
st.write("")  # spacer

base_kw = dict(ram=ram, weight=weight, ppi=ppi, cpu_ghz=cpu_ghz, ssd=ssd, hdd=hdd,
               inches=inches, touchscreen=touchscreen, ips=ips, company=company,
               type_name=type_name, cpu_tier=cpu_tier, gpu_brand=gpu_brand, os_choice=os_choice)

if predict_clicked:
    errors = []
    if ssd == 0 and hdd == 0:
        errors.append("A laptop needs some storage — set an SSD and/or HDD size above 0.")
    if os_choice == "Mac" and company != "Apple":
        errors.append("macOS is only sold on Apple laptops. Pick Apple as the brand, or a different OS.")

    if errors:
        for m in errors:
            st.warning(m)
    else:
        with st.spinner("Estimating price..."):
            try:
                price = predict_for(**base_kw)
            except Exception as e:
                st.error(f"Something went wrong during prediction: {e}")
                st.stop()
        price_text = f"S${price:,.2f}"
        lo, hi = max(0, price - 287), price + 287

        ## top row: animation and KPI cards with a gap column between them
        c1, gap, c2 = st.columns([1.1, 0.15, 1])
        with c1:
            components.html(laptop_animation_html(price_text), height=280)
        with c2:
            st.markdown(f'<div class="kpi"><div class="k-label">ESTIMATED PRICE</div>'
                        f'<div class="k-value">{price_text}</div></div>', unsafe_allow_html=True)
            k1, kgap, k2 = st.columns([1, 0.1, 1])
            k1.markdown(f'<div class="kpi"><div class="k-label">LOWER (−MAE)</div>'
                        f'<div class="k-value">S${lo:,.0f}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="kpi"><div class="k-label">UPPER (+MAE)</div>'
                        f'<div class="k-value">S${hi:,.0f}</div></div>', unsafe_allow_html=True)
            tier = "Budget" if price < 900 else ("Mid-range" if price < 1800 else "Premium")
            st.markdown(f'<div class="kpi"><div class="k-label">PRICE TIER</div>'
                        f'<div class="k-value">{tier}</div></div>', unsafe_allow_html=True)

        st.write("")
        st.success(f"Estimated price: {price_text}")
        st.write("")
        st.divider()
        st.write("")

        ## tabbed graphs
        t1, t2, t3 = st.tabs(["Price vs specs", "What drives this price", "Pricing range"])

        with t1:
            st.write("")
            g1, ggap, g2 = st.columns([1, 0.08, 1])
            with g1:
                ram_opts = [2,4,8,12,16,24,32,64]
                pr = [predict_for(**{**base_kw, "ram": r}) for r in ram_opts]
                fig, ax = plt.subplots(figsize=(5.5, 3.8)); fig.patch.set_alpha(0)
                ax.plot(ram_opts, pr, color=ACCENT, marker="o", lw=2.5)
                ax.axvline(ram, color="#888", ls="--", lw=1)
                style_ax(ax, "Price vs RAM", "RAM (GB)", "Predicted price (SGD)")
                plt.tight_layout(); st.pyplot(fig)
            with g2:
                ssd_opts = [0,128,256,512,1024]
                ps = [predict_for(**{**base_kw, "ssd": s}) for s in ssd_opts]
                fig, ax = plt.subplots(figsize=(5.5, 3.8)); fig.patch.set_alpha(0)
                ax.plot(ssd_opts, ps, color="#4dabf7", marker="o", lw=2.5)
                ax.axvline(ssd, color="#888", ls="--", lw=1)
                style_ax(ax, "Price vs SSD storage", "SSD (GB)", "Predicted price (SGD)")
                plt.tight_layout(); st.pyplot(fig)
            st.write("")
            st.caption("Each curve varies one spec while holding the others at your current "
                       "selection (dashed line = your choice). This shows which upgrades add the most value.")

        with t2:
            st.write("")
            try:
                imp = pd.Series(model.feature_importances_, index=feature_names)
                row = build_row(**base_kw)
                active = pd.Series(row)[imp.index]
                contrib = imp.copy().astype(float)
                contrib = contrib[[f for f in imp.index if active.get(f, 0) not in (0, 0.0)]]
                contrib = contrib.sort_values(ascending=True).tail(10)
                fig, ax = plt.subplots(figsize=(9, 4.2)); fig.patch.set_alpha(0)
                ax.barh(contrib.index, contrib.values, color=ACCENT)
                style_ax(ax, "Top factors influencing this laptop's price", "Model importance", "")
                plt.tight_layout(); st.pyplot(fig)
                st.write("")
                st.caption("Ranks the specs this laptop actually has by how much the model relies on "
                           "them overall — a plain-language view of what's pushing this price.")
            except Exception:
                st.info("Feature-contribution view is unavailable for this model type.")

        with t3:
            st.write("")
            fig, ax = plt.subplots(figsize=(9, 2.8)); fig.patch.set_alpha(0)
            bars = ax.barh(["Lower","Estimate","Upper"], [lo, price, hi],
                           color=["#9ec5fe", ACCENT, "#9ec5fe"])
            for b, v in zip(bars, [lo, price, hi]):
                ax.text(v + hi*0.01, b.get_y()+b.get_height()/2, f"S${v:,.0f}",
                        va="center", color=TEXT, fontsize=9)
            ax.set_xlim(0, hi*1.15)
            style_ax(ax, "Suggested pricing range (± model error)", "Price (SGD)", "")
            plt.tight_layout(); st.pyplot(fig)
            st.write("")
            st.caption("A practical price band: list around the estimate, using the range as room "
                       "to negotiate. Based on the model's typical error of about S$287.")

        st.session_state.history.append({
            "Brand": company, "Type": type_name, "RAM (GB)": ram,
            "Screen (in)": inches, "SSD (GB)": ssd, "Predicted price (SGD)": round(price, 2),
        })


## history
if st.session_state.history:
    st.write("")
    st.divider()
    st.write("")
    st.subheader("Prediction history")
    st.write("")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)
    st.write("")
    c1, c2 = st.columns([1, 5])
    c1.download_button("Download CSV", hist_df.to_csv(index=False).encode("utf-8"),
                       "prediction_history.csv", "text/csv")
    if c2.button("Clear history"):
        st.session_state.history = []
        st.rerun()
