# ==========================================
# FINAL CFST TRAINING CODE (SYNCED VERSION)
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_excel("CFSTtrainingdata.xlsx")

# Clean
df = df.dropna()

# Rename columns
df = df.rename(columns={
    "Bottom Diameter": "Diameter",
    "Height of Column": "Height",
    "Volume of Fibres": "Fibre",
    "Grade of Concrete": "Concrete",
    "Taper Angle": "Taper",
    "Axial Load Capacity": "Axial_Load",
    "Lateral Load Capacity Constant Amplitude": "Lateral_Constant",
    "Lateral Load Capacity Variable Amplitude": "Lateral_Variable"
})

# Convert concrete M30 → 30
# df["Concrete"] = df["Concrete"].astype(str).str.replace("M", "").astype(float)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

df["H_D"] = df["Height"] / df["Diameter"]
df["D_t"] = df["Diameter"] / df["Thickness"]
df["Slenderness"] = df["Height"] / df["Diameter"]
df["Thickness_Ratio"] = df["Thickness"] / df["Diameter"]
df["Fibre_Concrete"] = df["Fibre"] * df["Concrete"]

# ==========================================
# INPUT / OUTPUT
# ==========================================

X = df[[
    "Diameter", "Fibre", "Taper", "Thickness",
    "Concrete", "Height", "H_D", "D_t",
    "Slenderness", "Thickness_Ratio", "Fibre_Concrete"
]]

y = df[["Axial_Load", "Lateral_Constant", "Lateral_Variable"]]

#print("\nFinal Features Used:")
#print(X.columns)

# ==========================================
# SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# MODEL
# ==========================================

xgb = XGBRegressor(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)

model = MultiOutputRegressor(xgb)
model.fit(X_train, y_train)

# ==========================================
# EVALUATE
# ==========================================

y_pred = model.predict(X_test)

# Axial
print("Axial R2:", r2_score(y_test["Axial_Load"], y_pred[:, 0]))
print("Axial MAE:", mean_absolute_error(y_test["Axial_Load"], y_pred[:, 0]))

# Lateral Constant
print("Lateral Constant R2:", r2_score(y_test["Lateral_Constant"], y_pred[:, 1]))
print("Lateral Constant MAE:", mean_absolute_error(y_test["Lateral_Constant"], y_pred[:, 1]))

# Lateral Variable
print("Lateral Variable R2:", r2_score(y_test["Lateral_Variable"], y_pred[:, 2]))
print("Lateral Variable MAE:", mean_absolute_error(y_test["Lateral_Variable"], y_pred[:, 2]))

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "cfst_multi_model.pkl")

print("\n🎉 MODEL TRAINED & SAVED SUCCESSFULLY!")