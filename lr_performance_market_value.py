# check if general and position specific performace scores are 
# consistent with market value and transfer fee

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# read data
panel_pca = pd.read_csv("panel_with_all_pca.csv")

# group by position
position_groups = {
    'Attackers': ['Centre-Forward', 'Left Winger', 'Right Winger', 'Second Striker'],
    'Midfielders': ['Central Midfield', 'Defensive Midfield', 'Left Midfield', 'Right Midfield', 'Attacking Midfield'],
    'Defenders': ['Centre-Back', 'Left-Back', 'Right-Back'],
    'Goalkeepers': ['Goalkeeper']
}

# linear regression function
def analyze_market_value_regression(df, position_group):
    """
    特定のポジショングループに対して市場価値の回帰分析を実行
    """
    # extract relevant data for the position group
    pos_mask = df['position'].isin(position_groups[position_group])
    pos_df = df[pos_mask].copy()

    X = pos_df[['performance_composite_score', f'{position_group}_composite_score']]
    y = np.log(pos_df['market_value_in_eur'].replace(0, 1))
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    # linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # t値・p値を出す（statsmodels）
    X_sm = sm.add_constant(X)
    ols_model = sm.OLS(y, X_sm).fit()

    # results
    results = {
        'position_group': position_group,
        'sample_size': len(X),
        'r2_score': r2_score(y, model.predict(X)),
        'coefficients': dict(zip(X.columns, model.coef_)),
        'intercept': model.intercept_,
        'tvalues': {k: v for k, v in ols_model.tvalues.to_dict().items() if k != 'age'},
        'pvalues': {k: v for k, v in ols_model.pvalues.to_dict().items() if k != 'age'}
    }

    # summary
    print(f"\n[statsmodels summary for {position_group}]")
    print(ols_model.summary())

    return results

# conduct regression analysis for each position group
results = []
for position_group in position_groups.keys():
    result = analyze_market_value_regression(panel_pca, position_group)
    results.append(result)
    
    print(f"\nResults for {position_group}:")
    print(f"Sample size: {result['sample_size']}")
    print(f"R² score: {result['r2_score']:.3f}")
    print("\nCoefficients:")
    for k, v in result['coefficients'].items():
        print(f"  {k}: {v:.4f}")
    print("\nt-values:")
    for k, v in result['tvalues'].items():
        print(f"  {k}: {v:.4f}")
    print("\np-values:")
    for k, v in result['pvalues'].items():
        print(f"  {k}: {v:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("regression_results_market_value_by_position.csv", index=False)
print("\nSaved regression results to 'regression_results_market_value_by_position.csv'")