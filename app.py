import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# FUZZY FUNCTIONS
# ──────────────────────────────────────────────

def trapmf(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if b <= x <= c: return 1.0
    if x < b: return (x - a) / (b - a)
    return (d - x) / (d - c)

def get_membership(score):
    return {
        "A": trapmf(score, 83, 87, 100, 100),
        "B": trapmf(score, 67, 72,  82,  87),
        "C": trapmf(score, 57, 62,  67,  72),
        "D": trapmf(score, 47, 52,  57,  62),
        "E": trapmf(score,  0,  0,  47,  52),
    }

def attendance_bonus(att):
    att = float(np.clip(att, 0, 100))
    if att >= 90: return 10.0
    if att >= 60: return (att - 60) / 30 * 10
    return 0.0

def to_grade(score):
    if score > 85: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "E"

# ──────────────────────────────────────────────
# MAMDANI — defuzz centroid dari kurva kontinu
# ──────────────────────────────────────────────

def hitung_mamdani(assign, quiz, proj, mid, att):
    aq       = (assign + quiz) / 2
    weighted = aq * 0.20 + proj * 0.30 + mid * 0.40
    bonus    = attendance_bonus(att)
    final    = float(np.clip(weighted + bonus, 0, 100))
    mu       = get_membership(final)

    # Centroid dari output kurva yang di-clip oleh firing strength
    x_out = np.linspace(0, 100, 1000)
    agg   = np.zeros_like(x_out)
    funcs = {
        "A": lambda s: trapmf(s, 83, 87, 100, 100),
        "B": lambda s: trapmf(s, 67, 72,  82,  87),
        "C": lambda s: trapmf(s, 57, 62,  67,  72),
        "D": lambda s: trapmf(s, 47, 52,  57,  62),
        "E": lambda s: trapmf(s,  0,  0,  47,  52),
    }
    for g, fn in funcs.items():
        clipped = np.array([min(mu[g], fn(xi)) for xi in x_out])
        agg     = np.maximum(agg, clipped)

    defuzz = np.sum(x_out * agg) / np.sum(agg) if np.sum(agg) > 0 else final

    return {
        "method": "Mamdani",
        "aq": round(aq, 2), "weighted": round(weighted, 2),
        "bonus": round(bonus, 2), "final": round(final, 2),
        "mu": {g: round(v, 4) for g, v in mu.items()},
        "defuzz": round(defuzz, 2), "grade": to_grade(defuzz),
        "agg": agg, "x_out": x_out,
    }

# ──────────────────────────────────────────────
# SUGENO — output konstanta per rule
# ──────────────────────────────────────────────

SUGENO_K = {"A": 95, "B": 77, "C": 65, "D": 55, "E": 35}

def hitung_sugeno(assign, quiz, proj, mid, att):
    aq       = (assign + quiz) / 2
    weighted = aq * 0.20 + proj * 0.30 + mid * 0.40
    bonus    = attendance_bonus(att)
    final    = float(np.clip(weighted + bonus, 0, 100))
    mu       = get_membership(final)

    num    = sum(mu[g] * SUGENO_K[g] for g in mu)
    den    = sum(mu.values())
    defuzz = num / den if den != 0 else final

    return {
        "method": "Sugeno",
        "aq": round(aq, 2), "weighted": round(weighted, 2),
        "bonus": round(bonus, 2), "final": round(final, 2),
        "mu": {g: round(v, 4) for g, v in mu.items()},
        "defuzz": round(defuzz, 2), "grade": to_grade(defuzz),
    }

# ──────────────────────────────────────────────
# GRAFIK
# ──────────────────────────────────────────────

GCOLOR = {"A": "#639922", "B": "#378ADD", "C": "#EF9F27", "D": "#D85A30", "E": "#E24B4A"}

def plot_mamdani_membership(r):
    x   = np.linspace(0, 100, 500)
    funcs = {
        "A": lambda s: trapmf(s, 83, 87, 100, 100),
        "B": lambda s: trapmf(s, 67, 72,  82,  87),
        "C": lambda s: trapmf(s, 57, 62,  67,  72),
        "D": lambda s: trapmf(s, 47, 52,  57,  62),
        "E": lambda s: trapmf(s,  0,  0,  47,  52),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0E1117')

    # Kiri: semua kurva + garis skor
    ax = axes[0]
    ax.set_facecolor('#0E1117')
    for g, fn in funcs.items():
        y = np.array([fn(xi) for xi in x])
        ax.plot(x, y, color=GCOLOR[g], linewidth=2, label=f"Grade {g}")
        ax.fill_between(x, y, alpha=0.07, color=GCOLOR[g])
        val = fn(r["final"])
        if val > 0.01:
            ax.plot(r["final"], val, 'o', color=GCOLOR[g], markersize=7, zorder=5)

    ax.axvline(r["final"], color='white', linewidth=1.4, linestyle='--', alpha=0.7)
    ax.text(r["final"] + 1, 1.02, f"Skor {r['final']}", color='white', fontsize=9)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.15)
    ax.set_title("Fungsi Keanggotaan Input", color='white', fontsize=10)
    ax.set_xlabel("Skor", color='#aaa', fontsize=9)
    ax.set_ylabel("Derajat Keanggotaan μ", color='#aaa', fontsize=9)
    ax.tick_params(colors='#aaa'); ax.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='white', loc='upper left')
    ax.grid(axis='y', color='#333', linewidth=0.5)
    for sp in ax.spines.values(): sp.set_edgecolor('#444')

    # Kanan: output agregasi + defuzz
    ax2 = axes[1]
    ax2.set_facecolor('#0E1117')
    ax2.fill_between(r["x_out"], r["agg"], alpha=0.4, color='#378ADD')
    ax2.plot(r["x_out"], r["agg"], color='#378ADD', linewidth=1.5)
    ax2.axvline(r["defuzz"], color='#EF9F27', linewidth=2, linestyle='--')
    ax2.text(r["defuzz"] + 1, max(r["agg"]) * 0.85,
             f"Defuzz\n{r['defuzz']}", color='#EF9F27', fontsize=9)
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 1.1)
    ax2.set_title("Output Agregasi + Centroid (Mamdani)", color='white', fontsize=10)
    ax2.set_xlabel("Skor Output", color='#aaa', fontsize=9)
    ax2.set_ylabel("μ Agregasi", color='#aaa', fontsize=9)
    ax2.tick_params(colors='#aaa')
    ax2.grid(axis='y', color='#333', linewidth=0.5)
    for sp in ax2.spines.values(): sp.set_edgecolor('#444')

    plt.tight_layout()
    return fig


def plot_sugeno_membership(r):
    x = np.linspace(0, 100, 500)
    funcs = {
        "A": lambda s: trapmf(s, 83, 87, 100, 100),
        "B": lambda s: trapmf(s, 67, 72,  82,  87),
        "C": lambda s: trapmf(s, 57, 62,  67,  72),
        "D": lambda s: trapmf(s, 47, 52,  57,  62),
        "E": lambda s: trapmf(s,  0,  0,  47,  52),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0E1117')

    # Kiri: kurva membership
    ax = axes[0]
    ax.set_facecolor('#0E1117')
    for g, fn in funcs.items():
        y = np.array([fn(xi) for xi in x])
        ax.plot(x, y, color=GCOLOR[g], linewidth=2, label=f"Grade {g}")
        ax.fill_between(x, y, alpha=0.07, color=GCOLOR[g])
        val = fn(r["final"])
        if val > 0.01:
            ax.plot(r["final"], val, 'o', color=GCOLOR[g], markersize=7, zorder=5)

    ax.axvline(r["final"], color='white', linewidth=1.4, linestyle='--', alpha=0.7)
    ax.text(r["final"] + 1, 1.02, f"Skor {r['final']}", color='white', fontsize=9)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.15)
    ax.set_title("Fungsi Keanggotaan Input", color='white', fontsize=10)
    ax.set_xlabel("Skor", color='#aaa', fontsize=9)
    ax.set_ylabel("Derajat Keanggotaan μ", color='#aaa', fontsize=9)
    ax.tick_params(colors='#aaa'); ax.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='white', loc='upper left')
    ax.grid(axis='y', color='#333', linewidth=0.5)
    for sp in ax.spines.values(): sp.set_edgecolor('#444')

    # Kanan: output Sugeno — bar konstanta × firing strength
    ax2 = axes[1]
    ax2.set_facecolor('#0E1117')
    grades = list(SUGENO_K.keys())
    k_vals = [SUGENO_K[g] for g in grades]
    mu_vals = [r["mu"][g] for g in grades]
    bar_colors = [GCOLOR[g] for g in grades]

    bars = ax2.bar(grades, mu_vals, color=bar_colors, edgecolor='white', linewidth=0.8, width=0.5)
    for bar, kv, mv, g in zip(bars, k_vals, mu_vals, grades):
        ax2.text(bar.get_x() + bar.get_width()/2, mv + 0.01,
                 f"μ={mv:.3f}\nk={kv}", ha='center', va='bottom', color='white', fontsize=8)

    ax2.axhline(0, color='#555', linewidth=0.5)
    ax2.set_ylim(0, 1.3)
    ax2.set_title(f"Firing Strength per Rule (Sugeno)\nDefuzz = {r['defuzz']}", color='white', fontsize=10)
    ax2.set_xlabel("Grade", color='#aaa', fontsize=9)
    ax2.set_ylabel("Firing Strength μ", color='#aaa', fontsize=9)
    ax2.tick_params(colors='#aaa')
    ax2.grid(axis='y', color='#333', linewidth=0.5)
    for sp in ax2.spines.values(): sp.set_edgecolor('#444')

    plt.tight_layout()
    return fig


def plot_vs(hasil_list):
    names = [h["nama"] for h in hasil_list]
    mam   = [h["mamdani"]["defuzz"] for h in hasil_list]
    sug   = [h["sugeno"]["defuzz"]  for h in hasil_list]
    x     = np.arange(len(names))
    w     = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(names)*1.5), 4.5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    b1 = ax.bar(x - w/2, mam, w, label='Mamdani', color='#1D4E89', edgecolor='#378ADD', linewidth=1.2)
    b2 = ax.bar(x + w/2, sug, w, label='Sugeno',  color='#3B6D11', edgecolor='#639922', linewidth=1.2)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.7,
                f"{bar.get_height():.1f}", ha='center', va='bottom', color='white', fontsize=9)

    for val, lbl in [(85,'A'), (70,'B'), (60,'C'), (50,'D')]:
        ax.axhline(val, color='#555', linewidth=0.8, linestyle='--')
        ax.text(len(names) - 0.3, val + 0.5, lbl, color='#777', fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylim(0, 110)
    ax.set_ylabel("Skor Defuzzifikasi", color='#aaa', fontsize=10)
    ax.set_title("Mamdani vs Sugeno — Perbandingan Skor", color='white', fontsize=11, pad=10)
    ax.tick_params(colors='#aaa')
    ax.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=9)
    ax.grid(axis='y', color='#333', linewidth=0.5)
    for sp in ax.spines.values(): sp.set_edgecolor('#444')

    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────

st.set_page_config(page_title="Fuzzy Grade Calculator", page_icon="🎓", layout="centered")
st.title("🎓 Fuzzy Logic Grade Calculator")
st.caption("Mamdani & Sugeno · Tugas+Kuis 20% · Projek 30% · UTS 40% · Kehadiran bonus max +10")

n = st.number_input("Masukkan jumlah mahasiswa", min_value=1, max_value=50, value=2, step=1)
st.divider()

with st.form("input_form"):
    data_mhs = []
    for i in range(1, int(n) + 1):
        st.markdown(f"**Mahasiswa {i}**")
        nama = st.text_input("Nama", key=f"nama_{i}", placeholder=f"Nama mahasiswa {i}")
        c1, c2 = st.columns(2)
        with c1:
            assign = st.number_input(f"Avg Tugas (Mhs {i})",   0.0, 100.0, 75.0, key=f"assign_{i}")
            proj   = st.number_input(f"Skor Projek (Mhs {i})", 0.0, 100.0, 75.0, key=f"proj_{i}")
            att    = st.number_input(f"Kehadiran % (Mhs {i})", 0.0, 100.0, 80.0, key=f"att_{i}")
        with c2:
            quiz   = st.number_input(f"Avg Kuis (Mhs {i})",    0.0, 100.0, 75.0, key=f"quiz_{i}")
            mid    = st.number_input(f"Skor UTS (Mhs {i})",    0.0, 100.0, 75.0, key=f"mid_{i}")
        data_mhs.append({"nama": nama or f"Mahasiswa {i}",
                         "assign": assign, "quiz": quiz,
                         "proj": proj, "mid": mid, "att": att})
        st.markdown("---")
    submitted = st.form_submit_button("🔍 Hitung dengan Mamdani & Sugeno", use_container_width=True, type="primary")

if submitted:
    hasil_list = []
    for d in data_mhs:
        hasil_list.append({
            "nama":    d["nama"],
            "mamdani": hitung_mamdani(d["assign"], d["quiz"], d["proj"], d["mid"], d["att"]),
            "sugeno":  hitung_sugeno( d["assign"], d["quiz"], d["proj"], d["mid"], d["att"]),
        })

    grade_icons = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "E": "🔴"}
    grade_labels = ["A", "B", "C", "D", "E"]
    text_colors  = {"A": "#27500A", "B": "#0C447C", "C": "#633806", "D": "#712B13", "E": "#791F1F"}

    tab_mam, tab_sug, tab_vs = st.tabs(["📐 Mamdani", "📏 Sugeno", "⚖️ Perbandingan"])

    # ── TAB MAMDANI ──
    with tab_mam:
        st.markdown("### Metode Mamdani")
        st.caption("Defuzzifikasi: **Centroid** dari kurva output teragregasi (area-based)")

        count = {g: sum(1 for h in hasil_list if h["mamdani"]["grade"] == g) for g in grade_labels}
        cols = st.columns(5)
        for i, g in enumerate(grade_labels):
            with cols[i]:
                st.metric(f"{grade_icons[g]} Grade {g}", f"{count[g]} mhs")

        st.divider()
        for h in hasil_list:
            r = h["mamdani"]
            with st.container(border=True):
                ca, cb = st.columns([4, 1])
                with ca:
                    st.markdown(f"**{h['nama']}**")
                    st.caption(f"Subtotal 90%: {r['weighted']} | Bonus: +{r['bonus']} | Skor: {r['final']} | Defuzz: {r['defuzz']}")
                    for g in grade_labels:
                        v = float(r["mu"][g])
                        if v > 0:
                            st.progress(v, text=f"Grade {g}: {v:.4f}")
                with cb:
                    st.markdown(f"<div style='text-align:center;font-size:42px;font-weight:600;color:{text_colors[r['grade']]}'>{r['grade']}</div>", unsafe_allow_html=True)

            st.markdown(f"**📈 Grafik Mamdani — {h['nama']}**")
            fig = plot_mamdani_membership(r)
            st.pyplot(fig); plt.close(fig)
            st.markdown("---")

    # ── TAB SUGENO ──
    with tab_sug:
        st.markdown("### Metode Sugeno")
        st.caption("Defuzzifikasi: **Weighted Average** dari konstanta output tiap rule (singleton)")

        count = {g: sum(1 for h in hasil_list if h["sugeno"]["grade"] == g) for g in grade_labels}
        cols = st.columns(5)
        for i, g in enumerate(grade_labels):
            with cols[i]:
                st.metric(f"{grade_icons[g]} Grade {g}", f"{count[g]} mhs")

        st.divider()
        for h in hasil_list:
            r = h["sugeno"]
            with st.container(border=True):
                ca, cb = st.columns([4, 1])
                with ca:
                    st.markdown(f"**{h['nama']}**")
                    st.caption(f"Subtotal 90%: {r['weighted']} | Bonus: +{r['bonus']} | Skor: {r['final']} | Defuzz: {r['defuzz']}")
                    for g in grade_labels:
                        v = float(r["mu"][g])
                        if v > 0:
                            st.progress(v, text=f"Grade {g}: {v:.4f}")
                with cb:
                    st.markdown(f"<div style='text-align:center;font-size:42px;font-weight:600;color:{text_colors[r['grade']]}'>{r['grade']}</div>", unsafe_allow_html=True)

            st.markdown(f"**📈 Grafik Sugeno — {h['nama']}**")
            fig = plot_sugeno_membership(r)
            st.pyplot(fig); plt.close(fig)
            st.markdown("---")

    # ── TAB PERBANDINGAN ──
    with tab_vs:
        st.markdown("### Mamdani vs Sugeno")
        st.caption("Perbedaan utama: Mamdani menghasilkan kurva output → centroid. Sugeno menggunakan nilai konstanta → weighted average.")

        fig_vs = plot_vs(hasil_list)
        st.pyplot(fig_vs); plt.close(fig_vs)

        st.divider()
        st.markdown("**Ringkasan perbedaan hasil:**")
        for h in hasil_list:
            mam = h["mamdani"]
            sug = h["sugeno"]
            selisih = round(abs(mam["defuzz"] - sug["defuzz"]), 2)
            same = "✅ Sama" if mam["grade"] == sug["grade"] else f"⚠️ Beda: Mamdani={mam['grade']}, Sugeno={sug['grade']}"
            st.markdown(f"- **{h['nama']}** → Mamdani: `{mam['defuzz']}` | Sugeno: `{sug['defuzz']}` | Selisih: `{selisih}` | Grade: {same}")