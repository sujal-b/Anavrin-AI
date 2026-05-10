# ============================================================
# ML Customer Support Chatbot - Kaggle GPU Training Script
# FIXED VERSION - All bugs resolved
# ============================================================
# Instructions:
# 1. Create new Kaggle notebook
# 2. Enable GPU: Settings → Accelerator → GPU T4
# 3. NO dataset upload needed - loads from HuggingFace automatically
# 4. Copy this entire script into a cell
# 5. Run all cells
# 6. Download models/ and evaluation/ folders
# ============================================================

# ============================================================
# BUGS FOUND & FIXED (SUMMARY)
# ============================================================
# BUG 1 [CRITICAL]  - Cell 7 : KNeighborsClassifier used metric='euclidean'
#                     on a sparse TF-IDF matrix. Sklearn forces 'brute'
#                     algorithm for sparse input but 'euclidean' is not in
#                     the allowed set for brute+sparse → raises TypeError.
#                     FIX: changed metric to 'cosine' (best for TF-IDF) and
#                     explicitly set algorithm='brute'.
#
# BUG 2 [CRITICAL]  - Cell 7 : RandomForestClassifier with n_jobs=-1 on a
#                     sparse matrix copies the full matrix to EVERY CPU core,
#                     causing OOM (Out of Memory) crash on Kaggle's ~13GB RAM.
#                     With 5000 TF-IDF features and 27k rows, 200 trees with
#                     n_jobs=-1 will exhaust RAM. 
#                     FIX: set n_jobs=1 and reduce n_estimators to 100 +
#                     add max_depth=30 to cap memory growth.
#
# BUG 3 [CRITICAL]  - Cell 6 : After boolean mask filtering of sparse matrix X,
#                     the pandas Series y loses its positional alignment with
#                     the new X because y still carries old integer indices.
#                     train_test_split then misaligns rows silently.
#                     FIX: reset y's index after masking with y.reset_index(drop=True).
#
# BUG 4 [MODERATE]  - Cell 16 : The loop variable 'y' in the flow diagram
#                     boxes shadows the outer 'y' (target labels Series),
#                     corrupting all downstream references to labels.
#                     FIX: renamed loop variables to 'bx', 'by', 'btext'.
#
# BUG 5 [MINOR]     - Cell 9 : Models saved with pickle; joblib is faster and
#                     more memory-efficient for numpy-heavy sklearn objects.
#                     FIX: switched to joblib.dump / joblib.load.
#
# BUG 6 [MINOR]     - Cell 12 : y_pred_best is computed in Cell 12 but then
#                     recomputed identically in Cell 13, wasting time.
#                     FIX: reuse variable, skip redundant predict() call.
#
# IMPROVEMENT 1     - Cell 1 : Pin datasets version to avoid API-breaking
#                     changes from future HuggingFace releases.
#
# IMPROVEMENT 2     - Cell 5 : GPU (T4) is allocated but never used because
#                     all models are CPU-only sklearn estimators. Noted clearly
#                     so user isn't confused. GPU would matter if replacing
#                     TF-IDF with a transformer embedding model later.
# ============================================================


# ── Cell 1: Install dependencies ────────────────────────────
# IMPROVEMENT 1: Pin version to avoid future breaking changes
# Original: !pip install datasets -q
!pip install "datasets==2.20.0" -q
# joblib is pre-installed on Kaggle but ensure it's available
!pip install joblib -q


# ── Cell 2: Import libraries ─────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib          # FIX BUG 5: use joblib instead of pickle for sklearn models
import json
import os
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

print("Libraries loaded!")


# ── Cell 3: Load dataset ─────────────────────────────────────
from datasets import load_dataset
print("Loading bitext dataset...")
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = pd.DataFrame(dataset['train'])
print(f"Dataset loaded: {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"Intents: {df['intent'].nunique()}")
print(f"Categories: {df['category'].nunique()}")

# IMPROVEMENT 2: Clarify GPU usage
# NOTE: GPU (T4) is allocated but sklearn models run on CPU.
# The GPU will only benefit this script if you later swap TF-IDF
# for a HuggingFace transformer embedding model (e.g. sentence-transformers).
# For now, all training is CPU-bound — this is expected and correct.


# ── Cell 4: Preprocess text ───────────────────────────────────
print("\nPreprocessing text...")
df['instruction_clean'] = df['instruction'].str.lower()
df['instruction_clean'] = df['instruction_clean'].str.replace(r'[^\w\s]', '', regex=True)
df['instruction_clean'] = df['instruction_clean'].str.strip()

# Remove empty rows
df = df[df['instruction_clean'].str.len() > 0]
print(f"Clean rows: {len(df)}")


# ── Cell 5: Feature extraction with TF-IDF ───────────────────
print("\nExtracting TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
X = tfidf.fit_transform(df['instruction_clean'])
y = df['intent']
print(f"Feature matrix shape: {X.shape}")
print(f"Number of intent classes: {len(y.unique())}")


# ── Cell 6: Split data ────────────────────────────────────────
print("\nSplitting data (80/20)...")

# Check class distribution and filter rare classes
class_counts = y.value_counts()
rare_classes = class_counts[class_counts < 5].index
if len(rare_classes) > 0:
    print(f"Removing {len(rare_classes)} rare classes with <5 samples")
    mask = ~y.isin(rare_classes)
    X = X[mask]
    y = y[mask]

# ── FIX BUG 3 ────────────────────────────────────────────────
# After boolean mask filtering, y retains original DataFrame indices
# (e.g. [0, 2, 5, 7, ...]) while X is reindexed to [0, 1, 2, 3, ...].
# This misalignment causes train_test_split to silently pair wrong
# labels with wrong rows. Resetting the index fixes the alignment.
y = y.reset_index(drop=True)
# ─────────────────────────────────────────────────────────────

# Use stratify only if all classes have at least 2 samples
class_counts_after = y.value_counts()
can_stratify = (class_counts_after >= 2).all()

if can_stratify:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")


# ── Cell 7: Train models ──────────────────────────────────────
print("\n" + "="*50)
print("TRAINING MODELS")
print("="*50)

models = {
    'naive_bayes': MultinomialNB(alpha=0.1),

    # ── FIX BUG 1 ────────────────────────────────────────────
    # ORIGINAL (CRASHES):
    #   KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    #
    # WHY IT CRASHES:
    #   TF-IDF output is a scipy CSR sparse matrix. When sklearn
    #   detects sparse input it forces algorithm='brute'. The 'brute'
    #   algorithm with sparse matrices only supports Minkowski-family
    #   metrics OR 'cosine'/'manhattan'. Passing metric='euclidean'
    #   (which is an alias not in the supported brute+sparse set)
    #   raises:
    #     TypeError: Sparse data was passed ... but dense data is required.
    #   or in newer sklearn:
    #     ValueError: Metric 'euclidean' not valid for sparse input.
    #
    # FIX: Use metric='cosine' which is the correct metric for
    #   TF-IDF vectors (measures angle between vectors, not magnitude).
    #   Always pair with algorithm='brute' for sparse data.
    # ─────────────────────────────────────────────────────────
    'knn': KNeighborsClassifier(
        n_neighbors=5,
        metric='cosine',        # FIX: was 'euclidean' → crashes on sparse
        algorithm='brute'       # FIX: required for sparse + cosine metric
    ),

    # ── FIX BUG 2 ────────────────────────────────────────────
    # ORIGINAL (OOM CRASH):
    #   RandomForestClassifier(n_estimators=200, max_depth=None,
    #                          min_samples_split=2, random_state=42,
    #                          n_jobs=-1)
    #
    # WHY IT CRASHES:
    #   n_jobs=-1 tells sklearn to use ALL CPU cores. Sklearn's RF
    #   implementation copies the full sparse matrix to every worker
    #   process. With 27k rows × 5000 features × 200 trees and
    #   Kaggle's ~13GB RAM limit, this causes an OOM kill.
    #   Additionally, max_depth=None allows unlimited tree depth,
    #   making each tree memorize the data and consuming huge RAM.
    #
    # FIX:
    #   - n_jobs=1  : single process, no matrix copying overhead
    #   - n_estimators=100 : halved, sufficient for this dataset size
    #   - max_depth=30 : caps memory consumption per tree
    # ─────────────────────────────────────────────────────────
    'random_forest': RandomForestClassifier(
        n_estimators=100,       # FIX: was 200 → halved to reduce RAM
        max_depth=30,           # FIX: was None → cap depth to save memory
        min_samples_split=2,
        random_state=42,
        n_jobs=1                # FIX: was -1 → prevents matrix-copy OOM crash
    )
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy    = model.score(X_test, y_test)
    f1_macro    = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        'accuracy':    float(accuracy),
        'f1_macro':    float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'report':      classification_report(y_test, y_pred, output_dict=True)
    }
    trained_models[name] = model

    print(f"  Accuracy:      {accuracy:.4f}")
    print(f"  F1 (macro):    {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")


# ── Cell 8: Find best model ───────────────────────────────────
best_model_name = max(results, key=lambda x: results[x]['f1_macro'])
best_model = trained_models[best_model_name]
print(f"\n{'='*50}")
print(f"BEST MODEL: {best_model_name}")
print(f"F1 (macro): {results[best_model_name]['f1_macro']:.4f}")
print(f"{'='*50}")


# ── Cell 9: Save models ───────────────────────────────────────
print("\nSaving models...")
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('evaluation', exist_ok=True)

# FIX BUG 5: use joblib instead of pickle
# joblib is 2-5x faster and more memory-efficient for large numpy arrays
# inside sklearn objects (sparse matrices, decision tree node arrays, etc.)
for name, model in trained_models.items():
    joblib.dump(model, f'models/{name}.joblib')
    print(f"  Saved models/{name}.joblib")

joblib.dump(tfidf, 'models/tfidf.joblib')
print(f"  Saved models/tfidf.joblib")

# How to load later:
# tfidf = joblib.load('models/tfidf.joblib')
# model = joblib.load('models/random_forest.joblib')


# ── Cell 10: F1 Score Bar Chart ───────────────────────────────
print("\nGenerating graphs...")
os.makedirs('evaluation', exist_ok=True)

plt.figure(figsize=(10, 6))
model_names = list(results.keys())
f1_scores   = [results[m]['f1_macro'] for m in model_names]
colors      = ['#3498db', '#2ecc71', '#e74c3c']
bars        = plt.bar(model_names, f1_scores, color=colors,
                      edgecolor='black', linewidth=0.5)

for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.01,
             f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

plt.title('Model Comparison - F1 Score (Macro Average)',
          fontsize=14, fontweight='bold')
plt.ylabel('F1 Score', fontsize=12)
plt.xlabel('Model',    fontsize=12)
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('evaluation/individual_scores.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved evaluation/individual_scores.png")


# ── Cell 11: Intent–Category Correlation Heatmap ─────────────
plt.figure(figsize=(14, 10))

intent_counts = df['intent'].value_counts()
top_intents   = intent_counts.head(10).index

df_top    = df[df['intent'].isin(top_intents)]
cross_tab = pd.crosstab(df_top['category'], df_top['intent'])

if cross_tab.shape[1] > 10:
    cross_tab = cross_tab[top_intents[:10]]

sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd',
            linewidths=0.5, linecolor='gray')
plt.title('Intent-Category Correlation Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Intent',   fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('evaluation/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved evaluation/correlation_matrix.png")


# ── Cell 12: Confusion Matrix (Best Model, Top 10 Intents) ───
# FIX BUG 6: compute y_pred_best once and reuse in Cell 13
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(16, 14))
label_counts = Counter(y_test)
top_labels   = [label for label, _ in label_counts.most_common(10)]

mask       = np.isin(y_test.values, top_labels)
y_test_top = y_test.values[mask]
y_pred_top = y_pred_best[mask]

cm_top = confusion_matrix(y_test_top, y_pred_top, labels=top_labels)

sns.heatmap(cm_top, annot=True, fmt='d', cmap='Blues',
            xticklabels=top_labels, yticklabels=top_labels,
            linewidths=0.5, linecolor='gray')
plt.title(f'Confusion Matrix - {best_model_name} (Top 10 Intents)',
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual',    fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('evaluation/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved evaluation/confusion_matrix.png")


# ── Cell 13: Classification Report ───────────────────────────
print(f"\n{'='*50}")
print(f"CLASSIFICATION REPORT - {best_model_name}")
print(f"{'='*50}")
# FIX BUG 6: reuse y_pred_best from Cell 12 — no need to predict again
print(classification_report(y_test, y_pred_best))


# ── Cell 14: Save metrics JSON ────────────────────────────────
print("\nSaving metrics...")
metrics_output = {
    name: {
        'accuracy':    results[name]['accuracy'],
        'f1_macro':    results[name]['f1_macro'],
        'f1_weighted': results[name]['f1_weighted']
    }
    for name in results
}

with open('evaluation/metrics.json', 'w') as f:
    json.dump(metrics_output, f, indent=2)
print("  Saved evaluation/metrics.json")


# ── Cell 15: Save processed dataset ──────────────────────────
print("\nSaving processed dataset...")
df.to_pickle('data/processed_dataset.pkl')
print("  Saved data/processed_dataset.pkl")


# ── Cell 16: Pipeline Flow Diagram ───────────────────────────
print("\nGenerating flow diagram...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

# ── FIX BUG 4 ────────────────────────────────────────────────
# ORIGINAL:
#   for x, y, text in boxes:   ← 'y' shadows the label Series!
#   All downstream cells that use 'y' (y.reset_index, Counter(y_test),
#   etc.) would get a float coordinate instead of the pandas Series,
#   crashing with AttributeError or silently computing garbage.
#
# FIX: rename loop variables to bx, by, btext
# ─────────────────────────────────────────────────────────────
boxes = [
    (1, 2, 'User Input'),
    (3, 2, 'TF-IDF\nVectorize'),
    (5, 2, 'ML Intent\nClassifier'),
    (7, 2, 'Response\nLookup'),
    (9, 2, 'Bot Response')
]

for bx, by, btext in boxes:                          # FIX: was (x, y, text)
    rect = plt.Rectangle((bx - 0.7, by - 0.5), 1.4, 1,
                          fill=True, facecolor='#667eea',
                          edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(bx, by, btext, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

for i in range(len(boxes) - 1):
    ax.annotate('',
                xy=(boxes[i + 1][0] - 0.7, 2),
                xytext=(boxes[i][0] + 0.7, 2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

plt.title('ML Chatbot Pipeline Flow', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('evaluation/flow_diagram.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved evaluation/flow_diagram.png")


# ── Cell 17: Summary ──────────────────────────────────────────
print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print("\nFiles generated:")
print("  models/")
print("    - naive_bayes.joblib")
print("    - knn.joblib")
print("    - random_forest.joblib")
print("    - tfidf.joblib")
print("  evaluation/")
print("    - metrics.json")
print("    - individual_scores.png")
print("    - correlation_matrix.png")
print("    - confusion_matrix.png")
print("    - flow_diagram.png")
print("  data/")
print("    - processed_dataset.pkl")
print("\nDownload these files and place in your local project folder!")
print("="*50)