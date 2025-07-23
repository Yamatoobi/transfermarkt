import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ファイル名とタイトルの対応
files = [
    ("market_value_regression_coefficients.csv", "Market Value Regression Coefficients"),
    ("transfer_fee_regression_coefficients.csv", "Transfer Fee Regression Coefficients"),
    ("market_value_regression_coefficients_with_year.csv", "Market Value Regression Coefficients (with Year)"),
    ("transfer_fee_regression_coefficients_with_year.csv", "Transfer Fee Regression Coefficients (with Year)")
]

pdf_filename = "regression_results_summary.pdf"
with PdfPages(pdf_filename) as pdf:
    for fname, title in files:
        df = pd.read_csv(fname)
        plt.figure(figsize=(8, 5))
        plt.barh(df["feature"], df["coefficient"], color="skyblue")
        plt.xlabel("Coefficient")
        plt.title(f"{title}\nR² = {df['R2'].iloc[0]:.3f}")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        for i, v in enumerate(df["coefficient"]):
            plt.text(v, i, f"{v:.3f}", va="center", ha="left" if v > 0 else "right", color="black")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print(f"Saved summary PDF to {pdf_filename}")