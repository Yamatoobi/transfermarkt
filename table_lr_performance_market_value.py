import pandas as pd
import re

df = pd.read_csv("regression_results_market_value_by_position.csv")

def parse_dict_column(col):
    def parse_val(val):
        # 数値部分だけ抽出
        if isinstance(val, str):
            # 例: "{'performance_composite_score': np.float64(1.49223), ...}"
            # → {'performance_composite_score': 1.49223, ...}
            # まずnp.float64(...)を数値に変換
            val = re.sub(r"np\.float64\(([^)]+)\)", r"\1", val)
            # 次に辞書として評価
            return eval(val)
        return val
    return df[col].apply(parse_val)

df["coefficients_dict"] = parse_dict_column("coefficients")
df["tvalues_dict"] = parse_dict_column("tvalues")
df["pvalues_dict"] = parse_dict_column("pvalues")

rows = []
for _, row in df.iterrows():
    pos = row["position_group"]
    r2 = row["r2_score"]
    n = row["sample_size"]
    intercept = row["intercept"]
    coefs = row["coefficients_dict"]
    tvals = row["tvalues_dict"]
    pvals = row["pvalues_dict"]
    for key in coefs.keys():
        rows.append({
            "Position": pos,
            "Sample Size": n,
            "R²": round(r2, 3),
            "Intercept": round(intercept, 2),
            "Variable": key,
            "Coefficient": round(float(coefs[key]), 3),
            "t-value": round(float(tvals[key]), 2),
            "p-value": float(pvals[key])
        })

result_df = pd.DataFrame(rows)
print(result_df.to_string(index=False))