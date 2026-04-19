"""
=============================================================
03_data_appendix.py  –  Generate Data Appendix PDF
=============================================================
DS 4002  |  Group: Model Citizens
Members : Shaina Banduri, Neil Parikh, Nishana Dahal

PURPOSE
-------
Generates data/DataAppendix.pdf per the TIER Protocol 4.0 /
MI3 rubric requirements. Includes dataset description, unit
of observation, data dictionary, summary statistics, and
EDA figures embedded directly from output/.

USAGE
-----
Run from the repo root:
    python scripts/03_data_appendix.py

OUTPUT
------
    data/DataAppendix.pdf

REQUIREMENTS
------------
    pip install matplotlib Pillow
=============================================================
"""

import pathlib
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image as PILImage

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
REPO_ROOT  = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
DATA_DIR   = REPO_ROOT / "data"

# Portrait US Letter dimensions (inches)
PAGE_W, PAGE_H = 8.5, 11.0
L = 0.09   # left margin as fraction of page width
R = 0.91   # right margin as fraction of page width
TW = R - L # text width as fraction

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def new_page():
    """Return a blank portrait figure."""
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor('white')
    return fig


def txt(fig, x, y, s, **kwargs):
    """Shorthand for fig.text with common defaults."""
    kwargs.setdefault('ha', 'left')
    kwargs.setdefault('va', 'top')
    kwargs.setdefault('fontsize', 9.5)
    kwargs.setdefault('color', '#222222')
    fig.text(x, y, s, **kwargs)


def section(fig, y, title):
    """Render a bold section heading and return y position below it."""
    txt(fig, L, y, title, fontsize=12, fontweight='bold', color='#1a1a2e')
    return y - 0.028


def wrap(s, width=96):
    return textwrap.fill(s, width=width)


def render_table(fig, x, y, w, h, rows, headers, col_widths=None):
    """Draw a styled table inside the figure at the given axes rect."""
    ax = fig.add_axes([x, y, w, h])
    ax.axis('off')
    t = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='left',
        loc='center',
        colWidths=col_widths,
    )
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.55)
    for j in range(len(headers)):
        cell = t[0, j]
        cell.set_facecolor('#1a1a2e')
        cell.set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        bg = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            t[i, j].set_facecolor(bg)


def embed_image(fig, path, x, y, w, h):
    """Embed a PNG from disk as an image axis."""
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(PILImage.open(path))
    ax.axis('off')


# ------------------------------------------------------------------
# PAGE 1 – Title, Dataset Description, Unit of Observation,
#           Summary Statistics
# ------------------------------------------------------------------
def page_overview(pdf):
    fig = new_page()

    # Title block
    txt(fig, 0.5, 0.962, 'Data Appendix',
        ha='center', fontsize=20, fontweight='bold', color='#1a1a2e')
    txt(fig, 0.5, 0.930,
        'Classifying Leukemic B-Lymphoblast Cells from Normal B-Lymphoid Precursors\n'
        'Using Deep Learning on Microscopic Blood Smear Images',
        ha='center', fontsize=10, style='italic', color='#444444', linespacing=1.5)
    txt(fig, 0.5, 0.893,
        'Group: Model Citizens  |  Shaina Banduri, Neil Parikh, Nishana Dahal  |  DS 4002',
        ha='center', fontsize=9, color='#666666')

    # Rule
    ax_r = fig.add_axes([L, 0.880, TW, 0.003])
    ax_r.set_facecolor('#1a1a2e')
    ax_r.axis('off')

    # Section 1 – Dataset Description
    y = section(fig, 0.862, '1.  Dataset Description')
    desc = (
        'The C-NMC 2019 (Cancer vs. Normal Microscopic Cell) dataset was obtained from The Cancer '
        'Imaging Archive at the National Cancer Institute. It contains labeled microscopic blood smear '
        'cell images collected from 118 patients (69 cancer subjects, 49 normal subjects). Images were '
        'stained using the Jenner-Giemsa protocol and captured at 450×450 pixel resolution in BMP '
        'format. The dataset is pre-divided into three cross-validation folds for training and a '
        'separate held-out preliminary test set for final evaluation.'
    )
    txt(fig, L, y, wrap(desc), linespacing=1.55)
    y -= 0.095

    # Section 2 – Unit of Observation
    y = section(fig, y, '2.  Unit of Observation')
    unit = (
        'Each observation is a single microscopic blood smear cell image (one BMP file) associated '
        'with one patient. Each image carries a binary class label: 1 = leukemic B-lymphoblast '
        '(cancer / ALL) or 0 = normal B-lymphoid precursor (hem). The dataset contains no tabular '
        'covariates beyond the image file and its class label.'
    )
    txt(fig, L, y, wrap(unit), linespacing=1.55)
    y -= 0.078

    # Section 3 – Summary Statistics
    y = section(fig, y, '3.  Summary Statistics')

    rows = [
        ['Training — all folds combined', '10,661', '7,272  (68.2%)', '3,389  (31.8%)'],
        ['  fold_0', '3,527', '2,397  (67.9%)', '1,130  (32.1%)'],
        ['  fold_1', '3,581', '2,418  (67.5%)', '1,163  (32.5%)'],
        ['  fold_2 (validation split)', '3,553', '2,457  (69.2%)', '1,096  (30.8%)'],
        ['Test (held-out preliminary set)', '1,867', '1,219  (65.3%)', '648  (34.7%)'],
        ['Total', '12,528', '8,491  (67.8%)', '4,037  (32.2%)'],
    ]
    render_table(fig, L, y - 0.175, TW, 0.168, rows,
                 headers=['Split / Fold', 'Total Images', 'Cancer (all)', 'Normal (hem)'],
                 col_widths=[0.40, 0.17, 0.23, 0.23])
    y -= 0.195

    note = (
        'Note: A consistent ~2:1 class imbalance (cancer:normal) is present across all splits. '
        'This is addressed during training via inverse-frequency class weights in the cross-entropy loss.'
    )
    txt(fig, L, y, wrap(note), fontsize=8.5, color='#555555', style='italic', linespacing=1.4)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------
# PAGE 2 – Data Dictionary
# ------------------------------------------------------------------
def page_dictionary(pdf):
    fig = new_page()

    y = 0.955
    y = section(fig, y, '4.  Data Dictionary')

    intro = (
        'The dataset is organized as a directory of BMP image files. Class membership and fold '
        'assignment are encoded in the folder structure. The preliminary test set additionally '
        'provides a CSV file (C-NMC_test_prelim_phase_data_labels.csv) mapping each image filename '
        'to its ground-truth label.'
    )
    txt(fig, L, y, wrap(intro), linespacing=1.55)
    y -= 0.072

    # 4a – Training folder variables
    txt(fig, L, y, '4a.  Training Data — Folder-Structure Variables',
        fontsize=10.5, fontweight='semibold', color='#333333')
    y -= 0.022

    folder_rows = [
        ['fold', 'Cross-validation fold.', 'Categorical', 'fold_0 / fold_1 / fold_2'],
        ['class_folder', 'Subfolder encoding class label.', 'Categorical (binary)', 'all (cancer), hem (normal)'],
        ['label', 'Numeric label derived from class_folder.', 'Binary int', '1 = cancer, 0 = normal'],
        ['filename', 'BMP image filename (encodes patient/slide ID).', 'String', 'UID_57_29_1_all.bmp'],
        ['image', 'Raw microscopic blood smear cell image.', 'BMP, 450×450 px', '—'],
    ]
    render_table(fig, L, y - 0.145, TW, 0.138, folder_rows,
                 headers=['Variable', 'Description', 'Type', 'Example Values'],
                 col_widths=[0.18, 0.41, 0.21, 0.28])
    y -= 0.168

    # 4b – Test CSV variables
    txt(fig, L, y, '4b.  Test Set — CSV Variables  (C-NMC_test_prelim_phase_data_labels.csv)',
        fontsize=10.5, fontweight='semibold', color='#333333')
    y -= 0.022

    csv_rows = [
        ['Patient_ID', 'Original patient-level filename (encodes class).', 'String', 'UID_57_29_1_all.bmp'],
        ['new_names', 'Simplified numeric filename; used to locate image on disk.', 'String', '1.bmp, 4.bmp'],
        ['labels', 'Ground-truth binary class label.', 'Binary int', '1 = cancer, 0 = normal'],
    ]
    render_table(fig, L, y - 0.098, TW, 0.090, csv_rows,
                 headers=['Variable', 'Description', 'Type', 'Example Values'],
                 col_widths=[0.18, 0.41, 0.18, 0.28])
    y -= 0.118

    # 4c – Dataset metadata
    txt(fig, L, y, '4c.  Dataset Metadata',
        fontsize=10.5, fontweight='semibold', color='#333333')
    y -= 0.022

    meta_rows = [
        ['Cancer type', 'Type of cancer represented in positive class.', 'Constant', 'Acute Lymphoblastic Leukemia (ALL)'],
        ['Cancer location', 'Anatomical site.', 'Constant', 'Blood & Bone Marrow'],
        ['Staining protocol', 'Slide preparation procedure.', 'Constant', 'Jenner-Giemsa'],
        ['Native resolution', 'Raw BMP image size.', 'Integer (px)', '450 × 450'],
        ['Model input size', 'Resolution after preprocessing resize.', 'Integer (px)', '224 × 224'],
        ['Normalization', 'Per-channel pixel normalization statistics.', 'Float', 'ImageNet mean/std'],
        ['Source', 'Dataset provider.', 'String', 'The Cancer Imaging Archive (TCIA), NCI'],
        ['Last updated', 'Most recent dataset release date.', 'Date', '2019-05-28'],
    ]
    render_table(fig, L, y - 0.210, TW, 0.200, meta_rows,
                 headers=['Field', 'Description', 'Type', 'Value'],
                 col_widths=[0.20, 0.38, 0.18, 0.32])
    y -= 0.232

    # Section 5 – Current Unknowns (from MI2)
    y = section(fig, y, '5.  Current Unknowns')
    unknowns = (
        'It is unknown whether image characteristics differ systematically across patients or slides. '
        'Individual patients may exhibit unique visual signatures that could lead the model to learn '
        'patient-specific patterns instead of disease-relevant features. The fold-based split partially '
        'mitigates this risk by grouping images from the same patient into the same fold, but '
        'patient-level leakage cannot be fully ruled out without explicit patient-stratified '
        'cross-validation.'
    )
    txt(fig, L, y, wrap(unknowns), linespacing=1.55)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------
# PAGE 3 – EDA Figures: Q1 and Q2
# ------------------------------------------------------------------
def page_eda_q1_q2(pdf):
    fig = new_page()

    y = 0.955
    y = section(fig, y, '6.  Exploratory Data Analysis')

    txt(fig, L, y,
        'Three questions were explored during EDA. Figures are reproduced directly from output/.',
        fontsize=9, color='#555555')
    y -= 0.030

    # Q1
    txt(fig, L, y, 'Q1:  Is the training dataset class-balanced?',
        fontsize=10.5, fontweight='semibold', color='#333333')
    y -= 0.022
    q1 = (
        'No. The training set contains 7,272 cancer images (68.2%) and 3,389 normal images (31.8%), '
        'yielding an approximate 2:1 cancer-to-normal ratio that is consistent across all three folds. '
        'Inverse-frequency class weights were applied during training to compensate.'
    )
    txt(fig, L, y, wrap(q1), linespacing=1.55)
    y -= 0.058

    embed_image(fig, OUTPUT_DIR / 'q1_class_distribution.png',
                x=0.22, y=y - 0.255, w=0.56, h=0.255)
    y -= 0.268

    # Q2
    txt(fig, L, y, 'Q2:  How are images distributed across folds?',
        fontsize=10.5, fontweight='semibold', color='#333333')
    y -= 0.022
    q2 = (
        'Images are evenly distributed: fold_0 (n=3,527), fold_1 (n=3,581), fold_2 (n=3,553). '
        'The ~2:1 class imbalance is consistent within each fold, confirming a stratified split. '
        'fold_0 and fold_1 were used for training; fold_2 served as the validation set.'
    )
    txt(fig, L, y, wrap(q2), linespacing=1.55)
    y -= 0.058

    embed_image(fig, OUTPUT_DIR / 'q2_fold_distribution.png',
                x=0.16, y=y - 0.275, w=0.67, h=0.275)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------
# PAGE 4 – EDA Figure Q3 + References
# ------------------------------------------------------------------
def page_eda_q3_refs(pdf):
    fig = new_page()

    y = 0.955
    txt(fig, L, y, 'Q3:  Do cancer and normal images differ in average brightness?',
        fontsize=10.5, fontweight='semibold', color='#333333')
    y -= 0.022

    q3 = (
        'Yes. A sample of 150 images per class (300 total) was used to compare mean greyscale pixel '
        'brightness. Cancer images exhibit higher average brightness and a wider spread (roughly 7–15), '
        'while normal images are concentrated at lower brightness levels (roughly 3–8). This suggests '
        'greater stain uptake or morphological contrast in leukemic cells, consistent with known '
        'pathological characteristics of ALL.'
    )
    txt(fig, L, y, wrap(q3), linespacing=1.55)
    y -= 0.075

    embed_image(fig, OUTPUT_DIR / 'q3_brightness_by_class.png',
                x=0.12, y=y - 0.320, w=0.76, h=0.320)
    y -= 0.338

    # References
    y = section(fig, y, '7.  References')

    refs = [
        ('[1] American Cancer Society, "Key statistics for childhood leukemia," 2023. [Online]. '
         'Available: https://www.cancer.org/cancer/types/leukemia-in-children/key-statistics.html '
         '[Accessed: Mar. 30, 2026].'),
        ('[2] "C-NMC-2019," The Cancer Imaging Archive (TCIA). '
         'https://www.cancerimagingarchive.net/collection/c-nmc-2019/'),
        ('[3] S. J. Pan and Q. Yang, "A survey on transfer learning," IEEE Trans. Knowl. Data Eng., '
         'vol. 22, no. 10, pp. 1345–1359, Oct. 2010.'),
    ]
    for ref in refs:
        txt(fig, L, y, wrap(ref, width=94), fontsize=9, linespacing=1.45)
        y -= 0.052

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == '__main__':
    out_path = DATA_DIR / 'DataAppendix.pdf'
    with PdfPages(str(out_path)) as pdf:
        page_overview(pdf)
        page_dictionary(pdf)
        page_eda_q1_q2(pdf)
        page_eda_q3_refs(pdf)

        d = pdf.infodict()
        d['Title']   = 'Data Appendix — C-NMC 2019 Leukemia Classification'
        d['Author']  = 'Shaina Banduri, Neil Parikh, Nishana Dahal'
        d['Subject'] = 'DS 4002 Project 3 — Model Citizens'

    print(f'Saved: {out_path}')
