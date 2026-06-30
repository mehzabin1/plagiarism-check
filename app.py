import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Load student files ────────────────────────────────────────────────────────
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]


# ── Core logic ────────────────────────────────────────────────────────────────
def vectorize(texts):
    return TfidfVectorizer().fit_transform(texts).toarray()


def similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])[0][1]


vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()


def check_plagiarism():
    global s_vectors
    for student_a, vec_a in s_vectors:
        others = [pair for pair in s_vectors if pair[0] != student_a]
        for student_b, vec_b in others:
            sim_score = similarity(vec_a, vec_b)
            pair = sorted((student_a, student_b))
            plagiarism_results.add((pair[0], pair[1], sim_score))
    return plagiarism_results


results = check_plagiarism()

for data in results:
    print(data)


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_results(results, student_files):
    names = student_files                       # ordered list of file names
    n = len(names)

    # ── 1. Build symmetric similarity matrix ─────────────────────────────────
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 1.0)              # a doc is 100 % similar to itself

    name_index = {name: i for i, name in enumerate(names)}
    for a, b, score in results:
        i, j = name_index[a], name_index[b]
        matrix[i][j] = score
        matrix[j][i] = score

    short = [os.path.splitext(f)[0] for f in names]   # strip .txt for labels

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Plagiarism Detection Report', fontsize=15, fontweight='bold', y=1.02)

    # ── 2. Heatmap ────────────────────────────────────────────────────────────
    ax1 = axes[0]
    cmap = plt.cm.RdYlGn_r                     # green = low, red = high
    im = ax1.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(short, rotation=30, ha='right', fontsize=10)
    ax1.set_yticklabels(short, fontsize=10)
    ax1.set_title('Similarity Heatmap', fontsize=12, fontweight='bold')

    for i in range(n):
        for j in range(n):
            val = matrix[i][j]
            text_color = 'white' if val > 0.6 or val < 0.2 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=10, color=text_color, fontweight='bold')

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Similarity Score')

    # ── 3. Bar chart ─────────────────────────────────────────────────────────
    ax2 = axes[1]
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    pair_labels = [f'{os.path.splitext(a)[0]}\nvs\n{os.path.splitext(b)[0]}'
                   for a, b, _ in sorted_results]
    scores = [score for _, _, score in sorted_results]

    def bar_color(score):
        if score >= 0.7:
            return '#e74c3c'    # high – red
        elif score >= 0.4:
            return '#f39c12'    # medium – orange
        else:
            return '#2ecc71'    # low – green

    colors = [bar_color(s) for s in scores]
    bars = ax2.bar(pair_labels, scores, color=colors, edgecolor='white',
                   linewidth=0.8, width=0.5)

    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('Cosine Similarity Score', fontsize=10)
    ax2.set_title('Pairwise Similarity Scores', fontsize=12, fontweight='bold')
    ax2.axhline(0.7, color='#e74c3c', linestyle='--', linewidth=1,
                label='High-risk threshold (0.70)')
    ax2.axhline(0.4, color='#f39c12', linestyle='--', linewidth=1,
                label='Medium-risk threshold (0.40)')
    ax2.tick_params(axis='x', labelsize=8)

    for bar, score in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f'{score:.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    legend_patches = [
        mpatches.Patch(color='#e74c3c', label='High similarity  (≥ 0.70)'),
        mpatches.Patch(color='#f39c12', label='Medium similarity (0.40 – 0.69)'),
        mpatches.Patch(color='#2ecc71', label='Low similarity   (< 0.40)'),
    ]
    ax2.legend(handles=legend_patches, fontsize=8, loc='upper right')

    plt.tight_layout()
    output_path = 'plagiarism_report.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nChart saved → {output_path}')
    plt.show()


plot_results(results, student_files)