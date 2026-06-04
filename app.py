import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# KONSTANTA
# ──────────────────────────────────────────────────────────────────────────────

GRADE_LABELS  = ["A", "B", "C", "D", "E"]
GRADE_COLORS  = {"A": "#639922", "B": "#378ADD", "C": "#EF9F27", "D": "#D85A30", "E": "#E24B4A"}
GRADE_TEXT_FG = {"A": "#27500A", "B": "#0C447C", "C": "#633806", "D": "#712B13", "E": "#791F1F"}
GRADE_ICONS   = {"A": "🟢",      "B": "🔵",      "C": "🟡",      "D": "🟠",      "E": "🔴"}

SUGENO_K = {"A": 95, "B": 77, "C": 65, "D": 55, "E": 35}

# Fungsi keanggotaan trapezoid per grade
TRAP_PARAMS = {
    "A": (83,  87, 100, 100),
    "B": (67,  72,  82,  87),
    "C": (57,  62,  67,  72),
    "D": (47,  52,  57,  62),
    "E": ( 0,   0,  47,  52),
}

# Bobot komponen nilai
W_AQ   = 0.20   # Avg(Tugas + Kuis)
W_PROJ = 0.30   # Projek
W_MID  = 0.40   # UTS


# ──────────────────────────────────────────────────────────────────────────────
# FUNGSI FUZZY
# ──────────────────────────────────────────────────────────────────────────────

def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Fungsi keanggotaan trapezoid."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def get_membership(score: float) -> dict[str, float]:
    """Hitung derajat keanggotaan untuk semua grade."""
    return {g: trapmf(score, *TRAP_PARAMS[g]) for g in GRADE_LABELS}


def attendance_bonus(att: float) -> float:
    """Bonus kehadiran: max +10 untuk kehadiran >= 90%."""
    att = float(np.clip(att, 0, 100))
    if att >= 90:
        return 10.0
    if att >= 60:
        return (att - 60) / 30 * 10
    return 0.0


def hitung_skor_dasar(assign: float, quiz: float, proj: float, mid: float, att: float) -> dict:
    """Hitung skor terbobot sebelum defuzzifikasi."""
    aq       = (assign + quiz) / 2
    weighted = aq * W_AQ + proj * W_PROJ + mid * W_MID
    bonus    = attendance_bonus(att)
    final    = float(np.clip(weighted + bonus, 0, 100))
    return {"aq": round(aq, 2), "weighted": round(weighted, 2),
            "bonus": round(bonus, 2), "final": round(final, 2)}


def to_grade(score: float) -> str:
    if score > 85: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "E"


# ──────────────────────────────────────────────────────────────────────────────
# METODE MAMDANI
# ──────────────────────────────────────────────────────────────────────────────

def hitung_mamdani(assign: float, quiz: float, proj: float, mid: float, att: float) -> dict:
    """
    Defuzzifikasi Mamdani menggunakan metode centroid.
    Output berupa kurva agregasi kontinu yang di-clip oleh firing strength.
    """
    base = hitung_skor_dasar(assign, quiz, proj, mid, att)
    mu   = get_membership(base["final"])

    x_out = np.linspace(0, 100, 1000)
    agg   = np.zeros_like(x_out)

    for g in GRADE_LABELS:
        clipped = np.array([min(mu[g], trapmf(xi, *TRAP_PARAMS[g])) for xi in x_out])
        agg     = np.maximum(agg, clipped)

    denom  = np.sum(agg)
    defuzz = round(float(np.sum(x_out * agg) / denom) if denom > 0 else base["final"], 2)

    return {
        "method":   "Mamdani",
        **base,
        "mu":       {g: round(v, 4) for g, v in mu.items()},
        "defuzz":   defuzz,
        "grade":    to_grade(defuzz),
        "agg":      agg,
        "x_out":    x_out,
    }


# ──────────────────────────────────────────────────────────────────────────────
# METODE SUGENO
# ──────────────────────────────────────────────────────────────────────────────

def hitung_sugeno(assign: float, quiz: float, proj: float, mid: float, att: float) -> dict:
    """
    Defuzzifikasi Sugeno menggunakan weighted average dari konstanta output
    (singleton) per rule.
    """
    base = hitung_skor_dasar(assign, quiz, proj, mid, att)
    mu   = get_membership(base["final"])

    num    = sum(mu[g] * SUGENO_K[g] for g in GRADE_LABELS)
    den    = sum(mu.values())
    defuzz = round(num / den if den != 0 else base["final"], 2)

    return {
        "method":  "Sugeno",
        **base,
        "mu":      {g: round(v, 4) for g, v in mu.items()},
        "defuzz":  defuzz,
        "grade":   to_grade(defuzz),
    }


# ──────────────────────────────────────────────────────────────────────────────
# HELPER GRAFIK
# ──────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=11, pad=8, color="#111")
    ax.set_xlabel(xlabel, fontsize=9, color="#333")
    ax.set_ylabel(ylabel, fontsize=9, color="#333")
    ax.tick_params(colors="#333", labelsize=8)
    ax.set_ylim(-0.05, 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#aaa")
    ax.spines["bottom"].set_color("#aaa")
    ax.grid(axis="y", color="#ddd", linewidth=0.6, linestyle="--")


def _plot_membership_panel(ax, final: float, title: str):
    """Panel kiri: fungsi keanggotaan trapezoid + garis skor."""
    x = np.linspace(0, 100, 600)

    for g in GRADE_LABELS:
        y = np.array([trapmf(xi, *TRAP_PARAMS[g]) for xi in x])
        ax.plot(x, y, color="black", linewidth=1.2)
        peak_x = x[np.argmax(y)]
        if np.max(y) > 0.5:
            ax.text(peak_x, 1.05, g, ha="center", fontsize=9, color="#111", fontweight="bold")

    ax.axvline(final, color="red", linewidth=1.2, linestyle="--")
    ax.text(final + 0.5, 1.12, f"Skor: {final}", color="red", fontsize=8)

    for g in GRADE_LABELS:
        val = trapmf(final, *TRAP_PARAMS[g])
        if val > 0.01:
            ax.plot(final, val, "o", color="red", markersize=5, zorder=5)
            ax.vlines(final, 0, val, colors="red", linewidth=0.7, linestyle=":")

    ax.set_xlim(30, 100)
    _style_ax(ax, title, "Skor", "μ (Derajat Keanggotaan)")


# ──────────────────────────────────────────────────────────────────────────────
# GRAFIK MAMDANI
# ──────────────────────────────────────────────────────────────────────────────

def plot_mamdani(r: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("white")

    _plot_membership_panel(ax1, r["final"], "Fungsi Keanggotaan — Mamdani")

    # Panel kanan: kurva agregasi + centroid
    ax2.fill_between(r["x_out"], r["agg"], alpha=0.25, color="steelblue")
    ax2.plot(r["x_out"], r["agg"], color="black", linewidth=1.2)
    ax2.axvline(r["defuzz"], color="red", linewidth=1.5, linestyle="--")
    ax2.text(r["defuzz"] + 0.8, max(r["agg"]) * 0.85 + 0.02,
             f"Centroid\n{r['defuzz']}", color="red", fontsize=8)
    ax2.set_xlim(30, 100)
    _style_ax(ax2, "Output Agregasi + Centroid — Mamdani", "Skor Output", "μ Agregasi")

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# GRAFIK SUGENO
# ──────────────────────────────────────────────────────────────────────────────

def plot_sugeno(r: dict) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("white")

    _plot_membership_panel(ax1, r["final"], "Fungsi Keanggotaan — Sugeno")

    # Panel kanan: singleton output
    k_vals  = [SUGENO_K[g] for g in GRADE_LABELS]
    mu_vals = [r["mu"][g]  for g in GRADE_LABELS]

    for kv, mv, g in zip(k_vals, mu_vals, GRADE_LABELS):
        ax2.vlines(kv, 0, mv, colors="black", linewidth=2)
        ax2.plot(kv, mv, "o", color="black", markersize=6, zorder=5)
        ax2.text(kv, mv + 0.03, f"{g}\nμ={mv:.3f}", ha="center", fontsize=8, color="#111")

    ax2.axvline(r["defuzz"], color="red", linewidth=1.5, linestyle="--")
    ax2.text(r["defuzz"] + 0.5, 0.9, f"Defuzz\n{r['defuzz']}", color="red", fontsize=8)
    ax2.set_xlim(25, 105)
    ax2.set_xticks(k_vals)
    _style_ax(ax2, f"Singleton Output — Sugeno (WA = {r['defuzz']})", "Nilai Konstanta k", "Firing Strength μ")

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# GRAFIK PERBANDINGAN
# ──────────────────────────────────────────────────────────────────────────────

def plot_perbandingan(hasil_list: list[dict]) -> plt.Figure:
    names = [h["nama"]              for h in hasil_list]
    mam   = [h["mamdani"]["defuzz"] for h in hasil_list]
    sug   = [h["sugeno"]["defuzz"]  for h in hasil_list]
    x     = np.arange(len(names))
    w     = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.5), 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    b1 = ax.bar(x - w/2, mam, w, label="Mamdani", color="#AECBF0", edgecolor="#1D4E89", linewidth=1.2)
    b2 = ax.bar(x + w/2, sug, w, label="Sugeno",  color="#B5DDA4", edgecolor="#2E6B15", linewidth=1.2)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7,
                f"{bar.get_height():.1f}", ha="center", va="bottom", color="#111", fontsize=9)

    for threshold, label in [(85, "A"), (70, "B"), (60, "C"), (50, "D")]:
        ax.axhline(threshold, color="#bbb", linewidth=0.8, linestyle="--")
        ax.text(len(names) - 0.3, threshold + 0.5, label, color="#888", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Skor Defuzzifikasi", fontsize=10, color="#333")
    ax.set_title("Mamdani vs Sugeno — Perbandingan Skor", fontsize=11, pad=10, color="#111")
    ax.tick_params(colors="#333")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#aaa")
    ax.spines["bottom"].set_color("#aaa")
    ax.grid(axis="y", color="#ddd", linewidth=0.6, linestyle="--")

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# KOMPONEN UI
# ──────────────────────────────────────────────────────────────────────────────

def tampilkan_ringkasan_grade(hasil_list: list[dict], key: str):
    """Tampilkan metrik jumlah mahasiswa per grade."""
    count = {g: sum(1 for h in hasil_list if h[key]["grade"] == g) for g in GRADE_LABELS}
    cols  = st.columns(5)
    for i, g in enumerate(GRADE_LABELS):
        with cols[i]:
            st.metric(f"{GRADE_ICONS[g]} Grade {g}", f"{count[g]} mhs")


def tampilkan_kartu_mahasiswa(h: dict, key: str):
    """Tampilkan kartu hasil satu mahasiswa."""
    r = h[key]
    with st.container(border=True):
        col_info, col_grade = st.columns([4, 1])
        with col_info:
            st.markdown(f"**{h['nama']}**")
            st.caption(
                f"Subtotal 90%: {r['weighted']} | "
                f"Bonus: +{r['bonus']} | "
                f"Skor: {r['final']} | "
                f"Defuzz: {r['defuzz']}"
            )
            for g in GRADE_LABELS:
                v = float(r["mu"][g])
                if v > 0:
                    st.progress(v, text=f"Grade {g}: {v:.4f}")
        with col_grade:
            st.markdown(
                f"<div style='text-align:center;font-size:42px;"
                f"font-weight:600;color:{GRADE_TEXT_FG[r['grade']]}'>"
                f"{r['grade']}</div>",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Fuzzy Grade Calculator", page_icon="🎓", layout="centered")
    st.title("🎓 Fuzzy Logic Grade Calculator")
    st.caption("Mamdani & Sugeno · Tugas+Kuis 20% · Projek 30% · UTS 40% · Kehadiran bonus max +10")

    n = st.number_input("Masukkan jumlah mahasiswa", min_value=1, max_value=50, value=2, step=1)
    st.divider()

    # ── Form input mahasiswa
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
                quiz = st.number_input(f"Avg Kuis (Mhs {i})", 0.0, 100.0, 75.0, key=f"quiz_{i}")
                mid  = st.number_input(f"Skor UTS (Mhs {i})", 0.0, 100.0, 75.0, key=f"mid_{i}")
            data_mhs.append({
                "nama": nama or f"Mahasiswa {i}",
                "assign": assign, "quiz": quiz,
                "proj": proj, "mid": mid, "att": att,
            })
            st.markdown("---")

        submitted = st.form_submit_button(
            "🔍 Hitung dengan Mamdani & Sugeno",
            use_container_width=True,
            type="primary",
        )

    # ── Hitung & tampilkan hasil
    if not submitted:
        return

    hasil_list = [
        {
            "nama":    d["nama"],
            "mamdani": hitung_mamdani(d["assign"], d["quiz"], d["proj"], d["mid"], d["att"]),
            "sugeno":  hitung_sugeno( d["assign"], d["quiz"], d["proj"], d["mid"], d["att"]),
        }
        for d in data_mhs
    ]

    tab_mam, tab_sug, tab_vs = st.tabs(["📐 Mamdani", "📏 Sugeno", "⚖️ Perbandingan"])

    # ── Tab Mamdani
    with tab_mam:
        st.markdown("### Metode Mamdani")
        st.caption("Defuzzifikasi: **Centroid** dari kurva output teragregasi (area-based)")
        tampilkan_ringkasan_grade(hasil_list, "mamdani")
        st.divider()

        for h in hasil_list:
            tampilkan_kartu_mahasiswa(h, "mamdani")
            st.markdown(f"**📈 Grafik Mamdani — {h['nama']}**")
            fig = plot_mamdani(h["mamdani"])
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

    # ── Tab Sugeno
    with tab_sug:
        st.markdown("### Metode Sugeno")
        st.caption("Defuzzifikasi: **Weighted Average** dari konstanta output tiap rule (singleton)")
        tampilkan_ringkasan_grade(hasil_list, "sugeno")
        st.divider()

        for h in hasil_list:
            tampilkan_kartu_mahasiswa(h, "sugeno")
            st.markdown(f"**📈 Grafik Sugeno — {h['nama']}**")
            fig = plot_sugeno(h["sugeno"])
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")

    # ── Tab Perbandingan
    with tab_vs:
        st.markdown("### Mamdani vs Sugeno")
        st.caption(
            "Perbedaan utama: Mamdani menghasilkan kurva output → centroid. "
            "Sugeno menggunakan nilai konstanta → weighted average."
        )
        fig_vs = plot_perbandingan(hasil_list)
        st.pyplot(fig_vs)
        plt.close(fig_vs)

        st.divider()
        st.markdown("**Ringkasan perbedaan hasil:**")
        for h in hasil_list:
            mam     = h["mamdani"]
            sug     = h["sugeno"]
            selisih = round(abs(mam["defuzz"] - sug["defuzz"]), 2)
            grade_info = (
                "✅ Sama"
                if mam["grade"] == sug["grade"]
                else f"⚠️ Beda: Mamdani={mam['grade']}, Sugeno={sug['grade']}"
            )
            st.markdown(
                f"- **{h['nama']}** → "
                f"Mamdani: `{mam['defuzz']}` | "
                f"Sugeno: `{sug['defuzz']}` | "
                f"Selisih: `{selisih}` | "
                f"Grade: {grade_info}"
            )


if __name__ == "__main__":
    main()