import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable


def find_input_image():
    candidates = [
        "Fig5b_PyMOL_1AK4_QMaSIF.png",
        "Fig5b_PyMOL_1AK4_QMaSIF.jpg",
        "Fig5b_PyMOL_1AK4_QMaSIF.jpeg",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "فایل Fig5b_PyMOL_1AK4_QMaSIF (.png یا .jpg) در پوشه فعلی پیدا نشد."
    )


def tight_crop_white(img: np.ndarray, threshold: float = 0.97, pad: int = 24) -> np.ndarray:
    """برش دقیق حاشیه‌های سفید اضافی با حفظ حاشیه امن (pad)."""
    rgb = img[:, :, :3]
    non_white = np.any(rgb < threshold, axis=2)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]

    r0 = max(0, rows[0] - pad)
    r1 = min(img.shape[0], rows[-1] + pad)
    c0 = max(0, cols[0] - pad)
    c1 = min(img.shape[1], cols[-1] + pad)
    return img[r0:r1, c0:c1]


def build_publication_pymol_figure():
    in_path = find_input_image()
    print(f"[INFO] Loading PyMOL render: {in_path}")

    img = mpimg.imread(in_path)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    # اگر تصویر کانال آلفا (شفافیت) دارد، روی پس‌زمینه کاملاً سفید بنشان
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4]
        img = img[:, :, :3] * alpha + (1.0 - alpha)

    cropped = tight_crop_white(img, threshold=0.97, pad=28)
    h, w, _ = cropped.shape

    # ایجاد بوم با نسبت تصویر دقیق + جای مناسب برای Colorbar در پایین
    fig = plt.figure(figsize=(7.2, 6.4), dpi=300, facecolor="white")
    ax = fig.add_axes([0.04, 0.14, 0.92, 0.83])
    ax.imshow(cropped)
    ax.set_xlim(-0.24 * w, 1.02 * w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    # ۱. برچسب زنجیره هدف (راست‌چین در فضای سفید سمت چپ)
    ax.annotate(
        "Target Surface\n(Chain B)",
        xy=(0.27 * w, 0.17 * h),
        xytext=(0.13 * w, 0.11 * h),
        fontsize=9.5,
        fontweight="bold",
        color="#1E293B",
        ha="right",
        va="center",
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.4, connectionstyle="arc3,rad=-0.10"),
    )

    # ۲. برچسب نقطه داغ پیش‌بینی‌شده (راست‌چین و کاملاً بیرون از سطح پروتئین)
    ax.annotate(
        "Predicted\nBinding Hotspot",
        xy=(0.24 * w, 0.45 * h),
        xytext=(0.11 * w, 0.44 * h),
        fontsize=9.5,
        fontweight="bold",
        color="#B91C1C",
        ha="right",
        va="center",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.6, connectionstyle="arc3,rad=0.06"),
    )

    # ۳. برچسب زنجیره شریک واقعی (پایین-راست، فضای سفید کاملاً خالی)
    ax.annotate(
        "Experimental Partner\n(Chain C, Ground Truth)",
        xy=(0.35 * w, 0.72 * h),
        xytext=(0.56 * w, 0.70 * h),
        fontsize=9.5,
        fontweight="bold",
        color="#15803D",
        va="center",
        arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.5, connectionstyle="arc3,rad=0.12"),
    )

    # ۴. کادر خلاصه نتایج کمی واقعی 1AK4 (گوشه پایین-راست)
    stats_text = (
        "Unseen PDB Complex: 1AK4\n"
        "Full-Surface AUC = 83.09%\n"
        "Pearson r = −0.483 (p = 5.4×10⁻⁸)"
    )
    ax.text(
        0.97 * w,
        0.95 * h,
        stats_text,
        fontsize=8.5,
        color="#0F172A",
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            linewidth=1.0,
        ),
    )

    # ۵. نوار مقیاس رنگ دقیقاً منطبق با spectrum b, blue_white_red در PyMOL
    cax = fig.add_axes([0.16, 0.075, 0.68, 0.028])
    pymol_bwr = LinearSegmentedColormap.from_list("pymol_bwr", ["#0000FF", "#FFFFFF", "#FF0000"])
    sm = ScalarMappable(cmap=pymol_bwr, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.set_xticklabels(["Low (0.0)", "Mid (0.5)", "High (1.0)"], fontsize=9, fontweight="bold")
    cbar.set_label("Normalized QMaSIF Predicted Interface Score", fontsize=9.5, fontweight="bold", labelpad=5)
    cbar.outline.set_edgecolor("#64748B")
    cbar.outline.set_linewidth(0.8)

    out_path = "Fig5b_PyMOL_1AK4_Publication.png"
    plt.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close()
    print(f"✅ Publication-ready PyMOL figure saved to '{out_path}'")


if __name__ == "__main__":
    build_publication_pymol_figure()