# data_cleaner.py

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_age_ = None
        self.vip_true_mean_ = None
        self.vip_false_mean_ = None
        self.service_cols_ = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        self.columns_ = None  # Will store the final set of columns after training
    
    def fit(self, X, y=None):
        """
        Learn from training data (e.g., median age, VIP means),
        then determine final columns after transformations.
        """
        df = X.copy()
        
        # 1) Compute median Age
        self.median_age_ = df["Age"].median(skipna=True)
        if np.isnan(self.median_age_):
            self.median_age_ = 27  # Fallback if all are NaN
        
        # 2) Fill service columns with 0 for VIP means
        for col in self.service_cols_:
            df[col] = df[col].fillna(0)
        
        # 3) If VIP is entirely missing, fill with False
        if "VIP" not in df.columns or df["VIP"].notna().sum() == 0:
            df["VIP"] = False
        
        # 4) Compute average spending for VIP=True vs. VIP=False
        vip_true_mask = (df["VIP"] == True)
        vip_false_mask = (df["VIP"] == False)
        self.vip_true_mean_ = df.loc[vip_true_mask, self.service_cols_].mean(axis=1).mean()
        self.vip_false_mean_ = df.loc[vip_false_mask, self.service_cols_].mean(axis=1).mean()
        
        # 5) Transform to see final columns
        df_clean = self._transform_internal(df)
        
        # 6) Store those columns to match in transform
        self.columns_ = df_clean.columns
        
        return self
    
    def transform(self, X, y=None):
        """
        Apply cleaning steps to X, reindex columns to match training.
        """
        df = X.copy()
        df_clean = self._transform_internal(df)
        
        # Make sure test data has same columns as training
        df_clean = df_clean.reindex(columns=self.columns_, fill_value=0)
        
        return df_clean
    
    def _transform_internal(self, df):
        """
        The full cleaning process: splitting columns, 
        one-hot encoding, filling missing values, etc.
        """
        # ==================== 1) Split PassengerId ====================
        if "PassengerId" in df.columns:
            df["group"], df["pp"] = zip(*df["PassengerId"].apply(
                lambda x: re.split(r'_+', x) if isinstance(x, str) else ["0","0"]
            ))
            df.drop(columns=["PassengerId"], inplace=True)
        else:
            df["group"] = 0
            df["pp"] = 0

        df["group"] = pd.to_numeric(df["group"], errors="coerce").fillna(0).astype(int)
        df["pp"]    = pd.to_numeric(df["pp"],    errors="coerce").fillna(0).astype(int)
        
        # ==================== 2) Age Imputation ====================
        if "Age" in df.columns:
            df["Age"] = df["Age"].fillna(self.median_age_)
            df["Age"] = df["Age"].round().astype(int)
        else:
            df["Age"] = 0
        
        # ==================== 3) Name Handling ====================
        if "Name" in df.columns:
            df["Name"] = df["Name"].fillna("FirstName LastName")
            df["first_name"], df["last_name"] = zip(*df["Name"].apply(
                lambda x: re.split(r'\s+', x) if isinstance(x, str) else ["Unknown","Unknown"]
            ))
            df.drop(columns=["Name", "first_name"], errors="ignore", inplace=True)
        else:
            df["last_name"] = "Unknown"
        
        # ==================== 4) Service Cols => 0 if Missing ====================
        for col in self.service_cols_:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0)
        
        # ==================== 5) Cabin => Deck/Num/Side ====================
        if "Cabin" in df.columns:
            df["Cabin"] = df["Cabin"].fillna("Unknown/0/Unknown")
            df["Deck"], df["Num"], df["Side"] = zip(*df["Cabin"].apply(
                lambda x: re.split(r'/', x) if isinstance(x, str) else ["Unknown","0","Unknown"]
            ))
            df.drop(columns=["Cabin"], inplace=True)
        else:
            df["Deck"] = "Unknown"
            df["Num"]  = "0"
            df["Side"] = "Unknown"
        
        df = pd.get_dummies(df, columns=["Deck","Side"], drop_first=False)
        
        # ==================== 6) Destination & HomePlanet ====================
        if "Destination" in df.columns:
            if "HomePlanet" in df.columns:
                mask_dest_null = df["Destination"].isna()
                df.loc[mask_dest_null, "Destination"] = df.loc[mask_dest_null, "HomePlanet"]
            df["Destination"] = df["Destination"].fillna("Unknown")
        else:
            df["Destination"] = "Unknown"
        
        if "HomePlanet" not in df.columns:
            df["HomePlanet"] = "Unknown"
        else:
            df["HomePlanet"] = df["HomePlanet"].fillna("Unknown")
        
        df = pd.get_dummies(df, columns=["HomePlanet","Destination"], drop_first=False)
        
        # ==================== 7) CryoSleep Imputation ====================
        if "CryoSleep" not in df.columns:
            df["CryoSleep"] = False
        else:
            mask_service_used = (
                (df["RoomService"] != 0) |
                (df["FoodCourt"] != 0)  |
                (df["ShoppingMall"] != 0) |
                (df["Spa"] != 0)        |
                (df["VRDeck"] != 0)
            )
            q = df.query("CryoSleep.isnull()").index.intersection(df[mask_service_used].index)
            df.loc[q, "CryoSleep"] = False
            
            mask_service_none = (
                (df["RoomService"] == 0) &
                (df["FoodCourt"] == 0)  &
                (df["ShoppingMall"] == 0) &
                (df["Spa"] == 0)        &
                (df["VRDeck"] == 0)
            )
            v = df.query("CryoSleep.isnull()").index.intersection(df[mask_service_none].index)
            df.loc[v, "CryoSleep"] = True
            
            df["CryoSleep"] = df["CryoSleep"].fillna(False)
        
        # ==================== 8) VIP Imputation ====================
        if "VIP" not in df.columns:
            df["VIP"] = False
        
        mask_vip_nan = df["VIP"].isna()
        
        # (1) Age < 18 => VIP = False
        if "Age" in df.columns:
            df.loc[mask_vip_nan & (df["Age"] < 18), "VIP"] = False
        
        # (2) If Deck_G != 0 => VIP = False
        if "Deck_G" in df.columns:
            mask_deck_g = (df["Deck_G"] != 0)
            df.loc[mask_vip_nan & mask_deck_g, "VIP"] = False
        
        # (3) Compare row-average spending to self.vip_true_mean_ vs. vip_false_mean_
        mask_vip_nan = df["VIP"].isna()
        row_avg_spend = df.loc[mask_vip_nan, self.service_cols_].mean(axis=1)
        vip_true_diff  = (row_avg_spend - self.vip_true_mean_).abs()
        vip_false_diff = (row_avg_spend - self.vip_false_mean_).abs()
        df.loc[mask_vip_nan, "VIP"] = vip_true_diff < vip_false_diff
        
        df["VIP"] = df["VIP"].fillna(False)
        
        # ==================== 9) Convert Numeric/Bools to int ====================
        skip_cols = set(self.service_cols_ + ["last_name"])
        for col in df.columns:
            if col not in skip_cols:
                if df[col].dtype in [np.float64, np.int64, bool, object]:
                    try:
                        df[col] = df[col].astype(int)
                    except ValueError:
                        pass
        
         # >>> Drop 'last_name' right before returning df
        if "last_name" in df.columns:
            df.drop(columns=["last_name"], inplace=True)
        return df
