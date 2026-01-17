# Using the Comparison CSV for Paper Tables

## ✅ Generated CSV Format

The `runs_comparison.csv` contains one row per run with ALL metrics side-by-side:

```
trainer,dataset,image_size,epochs,batch_size,config,timestamp,run_dir,mAP_px,mAP_sp_max,mAUPRO_px,mAUROC_px,mAUROC_sp_max,...
```

This matches the format shown in benchmark papers (like Table 1 in your screenshot).

## 📊 Quick View in Spreadsheet

Open the CSV in Excel, Google Sheets, or Numbers for easy viewing and formatting:

```bash
# On Mac
open runs_comparison.csv

# Copy to clipboard for pasting into Google Sheets
cat runs_comparison.csv | pbcopy

# Or use LibreOffice
libreoffice runs_comparison.csv
```

## 🎯 Creating Publication Tables

### Method 1: Using Pandas (Python)

```python
import pandas as pd

# Read the CSV
df = pd.read_csv('runs_comparison.csv')

# Select key metrics for paper table (like your screenshot)
paper_metrics = [
    'trainer', 'dataset',
    'mAUROC_sp_max',  # Image-level AU-ROC
    'mAP_sp_max',     # Image-level AP
    'mF1_max_sp_max', # Image-level F1_max
    'mAUROC_px',      # Pixel-level AU-ROC
    'mAP_px',         # Pixel-level AP
    'mF1_max_px',     # Pixel-level F1_max
    'mAUPRO_px'       # Pixel-level AU-PRO
]

# Create clean table
paper_table = df[paper_metrics].copy()

# Round to 1 decimal place (like the paper)
for col in paper_table.columns:
    if col not in ['trainer', 'dataset']:
        paper_table[col] = paper_table[col].round(1)

# Sort by Image-level AUROC (descending)
paper_table = paper_table.sort_values('mAUROC_sp_max', ascending=False)

# Display
print(paper_table.to_string(index=False))

# Export to LaTeX for paper
print("\nLaTeX format:")
print(paper_table.to_latex(index=False))

# Export to CSV for easy import
paper_table.to_csv('paper_table.csv', index=False)
```

### Method 2: Direct Excel Formatting

1. Open `runs_comparison.csv` in Excel
2. Select columns you want (Trainer, Dataset, metrics)
3. Format numbers to 1 decimal place
4. Add conditional formatting (highlight best values in bold)
5. Sort by primary metric (e.g., mAUROC_sp_max)
6. Copy/paste into your paper

### Method 3: Using R

```r
library(tidyverse)
library(knitr)

# Read CSV
df <- read.csv('runs_comparison.csv')

# Select and format
paper_table <- df %>%
  select(trainer, dataset,
         mAUROC_sp_max, mAP_sp_max, mF1_max_sp_max,
         mAUROC_px, mAP_px, mF1_max_px, mAUPRO_px) %>%
  mutate(across(where(is.numeric), ~ round(., 1))) %>%
  arrange(desc(mAUROC_sp_max))

# View
print(paper_table)

# Export to LaTeX
kable(paper_table, format = "latex")
```

## 📈 Example: Recreating Your Screenshot Table

Based on your screenshot, here's how to create the exact format:

```python
import pandas as pd

df = pd.read_csv('runs_comparison.csv')

# Group by dataset if you have multiple
result_table = df.groupby(['dataset', 'trainer']).agg({
    'mAUROC_sp_max': 'mean',  # Image-level AU-ROC
    'mAP_sp_max': 'mean',     # Image-level AP
    'mF1_max_sp_max': 'mean', # Image-level F1_max
    'mAUROC_px': 'mean',      # Pixel-level AU-ROC
    'mAP_px': 'mean',         # Pixel-level AP
    'mF1_max_px': 'mean',     # Pixel-level F1_max
    'mAUPRO_px': 'mean'       # Pixel-level AU-PRO
}).round(1).reset_index()

# Pivot to get dataset as rows, methods as groups
result_table.columns = ['Dataset', 'Method', 'AU-ROC', 'AP', 'F1_max',
                        'AU-ROC', 'AP', 'F1_max', 'AU-PRO']

print(result_table)
```

## 🏆 Highlighting Best Results

### In Excel:
1. Select metric columns
2. Home → Conditional Formatting → Highlight Cell Rules → Top 10 Items
3. Or manually bold the best value in each column

### In Pandas:
```python
def highlight_best(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]

styled = paper_table.style.apply(highlight_best, subset=numeric_columns)
styled.to_excel('paper_table_formatted.xlsx')
```

### In LaTeX:
```python
# Find best values
best_values = {}
for col in numeric_cols:
    best_values[col] = df[col].max()

# Generate LaTeX with \textbf{} for best
def format_cell(value, col):
    if col in best_values and value == best_values[col]:
        return f"\\textbf{{{value:.1f}}}"
    else:
        return f"{value:.1f}"
```

## 📋 Column Name Mapping (for papers)

ADer metric → Paper format:
- `mAUROC_sp_max` → AU-ROC (Image-level)
- `mAP_sp_max` → AP (Image-level)
- `mF1_max_sp_max` → F1_max (Image-level)
- `mAUROC_px` → AU-ROC (Pixel-level)
- `mAP_px` → AP (Pixel-level)
- `mF1_max_px` → F1_max (Pixel-level)
- `mAUPRO_px` → AU-PRO (Pixel-level)

## 💡 Tips

1. **Remove incomplete runs**: Filter out rows with 0.000 values (incomplete tests)
   ```python
   df = df[df['mAUROC_sp_max'] > 0]
   ```

2. **Add standard deviation**: If you have multiple runs of same config
   ```python
   df.groupby('trainer').agg({'mAUROC_sp_max': ['mean', 'std']})
   ```

3. **Compare with baselines**: Add published results manually
   ```python
   baselines = pd.DataFrame({
       'trainer': ['RD4AD', 'UniAD', 'SimpleNet'],
       'mAUROC_sp_max': [94.6, 96.5, 95.3],
       ...
   })
   full_table = pd.concat([df, baselines])
   ```

4. **Export for Overleaf/LaTeX**:
   ```bash
   # Install dependencies
   pip install tabulate

   # Generate LaTeX
   python -c "
   import pandas as pd
   df = pd.read_csv('runs_comparison.csv')
   print(df.to_latex(index=False, float_format='%.1f'))
   " > results_table.tex
   ```

## 🎨 Example Output

Your comparison will look like:

```
Dataset    Method           Img AU-ROC  Img AP  Img F1  Px AU-ROC  Px AP  Px F1  Px AUPRO
-----------------------------------------------------------------------------------------
MVTec-AD   InvADTrainer        99.0     99.6    98.1    98.0      56.9   59.1   93.8
MVTec-AD   MAMBAADTrainer      97.6     99.2    97.1    97.4      55.5   58.3   93.5
MVTec-AD   UniADTrainer        97.6     99.2    97.3    96.9      44.8   50.1   90.7
MVTec-AD   SimpleNetTrainer    93.5     97.8    94.8    96.2      48.2   51.2   85.1
```

This matches the format in benchmark papers and can be directly copy-pasted or converted to LaTeX!
