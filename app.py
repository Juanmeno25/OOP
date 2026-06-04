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

# Input jumlah mahasiswa di luar form agar jumlah baris dinamis
n = st.number_input("Masukkan jumlah mahasiswa", min_value=1, max_value=50, value=2, step=1)

st.divider()

# Gunakan Form agar tidak re-run setiap kali mengetik
with st.form("input_data_form"):
    data_mahasiswa = []
    
    for i in range(1, int(n) + 1):
        st.markdown(f"### 📋 Mahasiswa {i}")
        nama = st.text_input("Nama", key=f"nama_{i}", placeholder=f"Nama mahasiswa {i}")

        col1, col2 = st.columns(2)
        with col1:
            assign = st.number_input(f"Avg Tugas (Mhs {i})", 0.0, 100.0, 75.0, key=f"assign_{i}")
            proj   = st.number_input(f"Skor Projek (Mhs {i})", 0.0, 100.0, 75.0, key=f"proj_{i}")
            att    = st.number_input(f"Kehadiran % (Mhs {i})", 0.0, 100.0, 80.0, key=f"att_{i}")
        with col2:
            quiz   = st.number_input(f"Avg Kuis (Mhs {i})", 0.0, 100.0, 75.0, key=f"quiz_{i}")
            mid    = st.number_input(f"Skor UTS (Mhs {i})", 0.0, 100.0, 75.0, key=f"mid_{i}")
        
        data_mahasiswa.append({
            "nama": nama or f"Mahasiswa {i}",
            "assign": assign, "quiz": quiz,
            "proj": proj, "mid": mid, "att": att,
        })
        st.markdown("---")

    # Tombol submit khusus di dalam form
    submitted = st.form_submit_button("🔍 Mulai Hitung Semua Grade", use_container_width=True, type="primary")

# ──────────────────────────────────────────────
# LOGIKA OUTPUT (Hanya muncul jika tombol ditekan)
# ──────────────────────────────────────────────

if submitted:
    hasil = []
    for d in data_mahasiswa:
        r = hitung_grade(d["assign"], d["quiz"], d["proj"], d["mid"], d["att"])
        hasil.append({"Nama": d["nama"], **r})

    st.subheader("📊 Rekap Hasil")

    # Ringkasan per Grade
    grade_labels = ["A", "B", "C", "D", "E"]
    grade_colors = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "E": "🔴"}
    count = {g: sum(1 for h in hasil if h["grade"] == g) for g in grade_labels}

    cols = st.columns(5)
    for i, g in enumerate(grade_labels):
        with cols[i]:
            st.metric(label=f"{grade_colors[g]} Grade {g}", value=f"{count[g]} mhs")

    st.divider()

    # Detail List Mahasiswa
    for h in hasil:
        grade = h["grade"]
        with st.container(border=True):
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"**{h['Nama']}**")
                st.caption(
                    f"Skor Akhir: **{h['final']}** | Defuzz: **{h['defuzz']}** | Bonus: +{h['bonus']}"
                )
                
                # Progress bars untuk derajat keanggotaan
                for g in grade_labels:
                    val = float(h["mu"][g])
                    if val > 0:
                        st.progress(val, text=f"Keanggotaan Grade {g}: {val:.4f}")

            with col_b:
                color = {'A': '#27500A', 'B': '#0C447C', 'C': '#633806', 'D': '#712B13', 'E': '#791F1F'}.get(grade, '#000')
                st.markdown(
                    f"<div style='text-align:center; font-size:42px; font-weight:600; color:{color};'>{grade}</div>", 
                    unsafe_allow_html=True
                )

    # Tabel Ringkasan Akhir
    st.subheader("📋 Tabel Ringkasan")
    tabel = [{
        "Nama": h["Nama"],
        "Skor Akhir": h["final"],
        "Defuzz": h["defuzz"],
        "Grade": h["grade"],
    } for h in hasil]
    st.dataframe(tabel, use_container_width=True, hide_index=True)