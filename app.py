import streamlit as st
import numpy as np

# ──────────────────────────────────────────────
# FUZZY FUNCTIONS
# ──────────────────────────────────────────────

def trapmf(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
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
    if att >= 90:
        return 10.0
    elif att >= 60:
        return (att - 60) / 30 * 10
    return 0.0

def defuzzify(mu):
    centers = {"A": 92, "B": 77, "C": 64, "D": 54, "E": 40}
    num = sum(mu[g] * centers[g] for g in mu)
    den = sum(mu.values())
    return num / den if den != 0 else 0.0

def to_grade(score):
    if score > 85: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "E"

def hitung_grade(assign, quiz, proj, mid, att):
    aq      = (assign + quiz) / 2
    weighted = aq * 0.20 + proj * 0.30 + mid * 0.40
    bonus   = attendance_bonus(att)
    final   = float(np.clip(weighted + bonus, 0, 100))
    mu      = get_membership(final)
    defuzz  = defuzzify(mu)
    return {
        "aq":       round(aq, 2),
        "weighted": round(weighted, 2),
        "bonus":    round(bonus, 2),
        "final":    round(final, 2),
        "mu":       {g: round(v, 4) for g, v in mu.items()},
        "defuzz":   round(defuzz, 2),
        "grade":    to_grade(defuzz),
    }

# ──────────────────────────────────────────────
# STREAMLIT APP
# ──────────────────────────────────────────────

st.set_page_config(page_title="Fuzzy Grade Calculator", page_icon="🎓", layout="centered")

st.title("🎓 Fuzzy Logic Grade Calculator")
st.caption("Tugas+Kuis 20% · Projek 30% · UTS 40% · Kehadiran bonus max +10")

st.divider()

# ── Jumlah mahasiswa ──
n = st.number_input("Masukkan jumlah mahasiswa", min_value=1, max_value=50, value=2, step=1)

st.divider()

# ── Input form tiap mahasiswa ──
data_mahasiswa = []

for i in range(1, int(n) + 1):
    with st.expander(f"📋 Mahasiswa {i}", expanded=True):
        nama = st.text_input("Nama", key=f"nama_{i}", placeholder="Nama mahasiswa")

        col1, col2 = st.columns(2)
        with col1:
            assign = st.number_input("Assignments Avg (0–100)", 0.0, 100.0, 75.0, 0.1, key=f"assign_{i}")
            proj   = st.number_input("Projects Score (0–100)",  0.0, 100.0, 75.0, 0.1, key=f"proj_{i}")
            att    = st.number_input("Attendance (%)(0–100)",   0.0, 100.0, 80.0, 0.1, key=f"att_{i}")
        with col2:
            quiz   = st.number_input("Quizzes Avg (0–100)",     0.0, 100.0, 75.0, 0.1, key=f"quiz_{i}")
            mid    = st.number_input("Midterm Score (0–100)",   0.0, 100.0, 75.0, 0.1, key=f"mid_{i}")

        data_mahasiswa.append({
            "nama": nama or f"Mahasiswa {i}",
            "assign": assign, "quiz": quiz,
            "proj": proj, "mid": mid, "att": att,
        })

st.divider()

# ── Tombol hitung ──
if st.button("🔍 Hitung Semua Grade", use_container_width=True, type="primary"):

    hasil = []
    for d in data_mahasiswa:
        r = hitung_grade(d["assign"], d["quiz"], d["proj"], d["mid"], d["att"])
        hasil.append({"Nama": d["nama"], **r})

    st.subheader("📊 Rekap Hasil")

    # Summary grade count
    grade_labels = ["A", "B", "C", "D", "E"]
    grade_colors = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "E": "🔴"}
    count = {g: sum(1 for h in hasil if h["grade"] == g) for g in grade_labels}

    cols = st.columns(5)
    for i, g in enumerate(grade_labels):
        with cols[i]:
            st.metric(label=f"{grade_colors[g]} Grade {g}", value=f"{count[g]} mhs")

    st.divider()

    # Detail tiap mahasiswa
    for h in hasil:
        grade = h["grade"]
        color_map = {"A": "green", "B": "blue", "C": "orange", "D": "red", "E": "red"}

        with st.container(border=True):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"**{h['Nama']}**")
                st.caption(
                    f"Tugas+Kuis avg: {h['aq']} → ×20% = {round(h['aq']*0.20,2)}  |  "
                    f"Projek ×30% = {round(h['proj']*0.30 if 'proj' not in h else 0,2)}  |  "
                    f"UTS ×40%  |  Subtotal: **{h['weighted']}**  |  "
                    f"Bonus kehadiran: +{h['bonus']}  |  Skor akhir: **{h['final']}**"
                )

                # Membership bars
                st.markdown("**Derajat keanggotaan:**")
                for g in grade_labels:
                    st.progress(float(h["mu"][g]), text=f"Grade {g}: {h['mu'][g]:.4f}")

            with col_b:
                st.markdown(
                    f"<div style='text-align:center; font-size:42px; font-weight:600; "
                    f"padding-top:10px; color:{'#27500A' if grade=='A' else '#0C447C' if grade=='B' else '#633806' if grade=='C' else '#712B13' if grade=='D' else '#791F1F'}'>"
                    f"{grade}</div>"
                    f"<div style='text-align:center; font-size:13px; color:#888'>{h['defuzz']}</div>",
                    unsafe_allow_html=True
                )

    st.divider()

    # Tabel ringkasan
    st.subheader("📋 Tabel Ringkasan")
    tabel = [{
        "Nama":       h["Nama"],
        "Subtotal":   h["weighted"],
        "Bonus":      h["bonus"],
        "Skor Akhir": h["final"],
        "Defuzz":     h["defuzz"],
        "Grade":      h["grade"],
    } for h in hasil]
    st.dataframe(tabel, use_container_width=True, hide_index=True)  