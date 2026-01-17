# How to Compare Training Runs in ADer

## ✅ NEW: Single CSV Table Format

The comparison script now **automatically generates a single CSV file** with all metrics side-by-side in one table - perfect for creating benchmark tables like those in papers!

## Quick Start

### 1. Generate Comparison CSV

```bash
# Automatically creates runs_comparison.csv
python compare_runs.py

# With custom output
python compare_runs.py --output my_results.csv

# Show only top-3 in console (CSV still has all runs)
python compare_runs.py --top_k 3
```

### 2. Convert to Paper-Ready Format

```bash
# Clean CSV for papers (removes incomplete runs, better column names)
python csv_to_paper_table.py

# Generate LaTeX table for your paper
python csv_to_paper_table.py --format latex --output table.tex

# Generate Markdown table
python csv_to_paper_table.py --format markdown
```

## 📊 Output Files

### runs_comparison.csv (Raw Data)
**Complete data** - one row per run with ALL metrics:
- Metadata: trainer, dataset, image_size, epochs, batch_size, config, timestamp, run_dir
- All metrics: mAP_px, mAP_sp_max, mAUPRO_px, mAUROC_px, mAUROC_sp_max, etc.

**Example:**
```csv
trainer,dataset,image_size,epochs,mAUROC_sp_max,mAUROC_px,mAUPRO_px,...
InvADTrainer,mvtec,256,300,98.972,98.045,93.826,...
MAMBAADTrainer,mvtec,256,100,97.618,97.423,93.489,...
```

### paper_table.csv (Clean Format)
**Publication-ready** - filtered and formatted for papers:
- Removes incomplete runs
- Better column names (Img-AUROC, Px-AUROC, etc.)
- Sorted by performance
- Rounded to 1 decimal

**Example:**
```
Method,Dataset,Img-AUROC,Img-AP,Img-F1,Px-AUROC,Px-AP,Px-F1,Px-AUPRO
InvADTrainer,mvtec,99.0,99.6,98.1,98.0,56.9,59.1,93.8
MAMBAADTrainer,mvtec,97.6,99.2,97.1,97.4,55.5,58.3,93.5
```

## 📈 Using in Papers

### Option 1: Direct Import to Excel/Google Sheets

```bash
# Open in Excel (Mac)
open paper_table.csv

# Open in Excel (Windows)
start paper_table.csv

# Copy to clipboard
cat paper_table.csv | pbcopy  # Mac
cat paper_table.csv | xclip    # Linux
```

Then:
1. Paste into your document
2. Bold the best values
3. Add to your paper

### Option 2: LaTeX (Academic Papers)

```bash
# Generate LaTeX table
python csv_to_paper_table.py --format latex --output table.tex
```

Then in your paper:
```latex
\begin{table}[t]
\centering
\caption{Quantitative Results on MVTec-AD for multi-class setting.}
\label{tab:results}
\input{table.tex}
\end{table}
```

### Option 3: Pandas Analysis

```python
import pandas as pd

# Read the CSV
df = pd.read_csv('runs_comparison.csv')

# Filter complete runs
df = df[df['mAUROC_sp_max'] > 0]

# Get top 3 methods
top3 = df.nlargest(3, 'mAUROC_sp_max')

# Compare with published baselines
baselines = pd.DataFrame({
    'trainer': ['RD4AD', 'UniAD', 'DiAD'],
    'mAUROC_sp_max': [94.6, 96.5, 97.2],
    'mAUROC_px': [96.1, 96.8, 96.8],
    'mAUPRO_px': [91.1, 90.7, 90.7]
})

combined = pd.concat([df, baselines]).sort_values('mAUROC_sp_max', ascending=False)
print(combined[['trainer', 'mAUROC_sp_max', 'mAUROC_px', 'mAUPRO_px']])
```

## 🎯 Key Metrics Explained

### Image-Level (Detection)
- **Img-AUROC** (`mAUROC_sp_max`): How well the model detects anomalous images
- **Img-AP** (`mAP_sp_max`): Average Precision for image classification
- **Img-F1** (`mF1_max_sp_max`): F1 score (balance of precision/recall)

### Pixel-Level (Localization)
- **Px-AUROC** (`mAUROC_px`): How well the model localizes anomalies
- **Px-AP** (`mAP_px`): Average Precision for segmentation
- **Px-F1** (`mF1_max_px`): F1 score for segmentation
- **Px-AUPRO** (`mAUPRO_px`): ⭐ **Recommended** - Per-Region Overlap (more robust)

## 💡 Example Workflow

```bash
# 1. Train your models
CUDA_VISIBLE_DEVICES=0 python run.py -c configs.benchmark.simplenet.simplenet_256_100e -m train
CUDA_VISIBLE_DEVICES=0 python run.py -c configs.benchmark.mambaad.mambaad_256_100e -m train
CUDA_VISIBLE_DEVICES=0 python run.py -c configs.benchmark.mamba_clip.mamba_clip_mvtec_518 -m train

# 2. Generate comparison
python compare_runs.py

# 3. Create paper table
python csv_to_paper_table.py --format latex

# 4. Open CSV in Excel for detailed analysis
open runs_comparison.csv

# 5. Visualize with TensorBoard
tensorboard --logdir runs/
```

## 📋 Console Output

The script also prints a nice console summary:

```
========================================================================================================================
SUMMARY TABLE
========================================================================================================================

Trainer              Dataset    Size   Epochs  Img-AUROC  Px-AUROC   Px-AUPRO
-------------------------------------------------------------------------------
InvADTrainer         mvtec      256    300        99.0      98.0      93.8
MAMBAADTrainer       mvtec      256    100        97.6      97.4      93.5
UniADTrainer         mvtec      256    1000       97.6      96.9      90.7

========================================================================================================================
BEST MODELS
========================================================================================================================

🏆 mAUROC_sp_max:
   Value:   98.972
   Trainer: InvADTrainer
   ...
```

## 🎨 Advanced: Custom Tables

For custom metric selection:

```python
import pandas as pd

df = pd.read_csv('runs_comparison.csv')

# Your custom selection
custom_metrics = ['trainer', 'dataset', 'epochs',
                  'mAUROC_sp_max', 'mAUPRO_px', 'mIoU_max_px']

custom_table = df[custom_metrics].copy()
custom_table = custom_table[custom_table['mAUROC_sp_max'] > 0]
custom_table = custom_table.round(1).sort_values('mAUROC_sp_max', ascending=False)

print(custom_table.to_markdown(index=False))
```

## 📚 See Also

- **USING_CSV_FOR_PAPERS.md** - Detailed guide on creating publication tables
- **compare_runs.py** - Main comparison script
- **csv_to_paper_table.py** - Convert to paper-ready format

## ❓ FAQ

**Q: Can I compare runs from different datasets?**
A: Yes! The CSV includes a dataset column. Filter by dataset when analyzing.

**Q: How do I add my own baseline results?**
A: Add rows to the CSV manually or use pandas to merge with published results.

**Q: The values show 0.000 for some runs**
A: Those runs haven't completed testing yet. Filter them out:
```python
df = df[df['mAUROC_sp_max'] > 0]
```

**Q: Can I highlight the best values automatically?**
A: In Excel: Conditional Formatting → Highlight Cell Rules
In Python: Use `.style.highlight_max()` or see USING_CSV_FOR_PAPERS.md

**Q: How do I update the comparison after new runs?**
A: Just run `python compare_runs.py` again - it regenerates the CSV with all runs including new ones.
