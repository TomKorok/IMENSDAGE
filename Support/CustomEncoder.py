import pandas as pd


class CustomEncoder:
    def __init__(self):
        self.encoders = {}
        self.fitted = False

    def fit(self, X):
        """
        Fit the encoder to unique labels for each column.
        """
        if isinstance(X, pd.Series):
            unique_classes = sorted(X.dropna().unique())
            self.encoders[0] = {
                'classes_': {label: idx for idx, label in enumerate(unique_classes)},
                'inverse_classes_': {idx: label for label, idx in {
                    label: idx for idx, label in enumerate(unique_classes)
                }.items()},
                'min_idx': 0,
                'max_idx': len(unique_classes) - 1
            }
        elif isinstance(X, pd.DataFrame):
            for col in X.columns:
                unique_classes = sorted(X[col].dropna().unique())
                self.encoders[col] = {
                    'classes_': {label: idx for idx, label in enumerate(unique_classes)},
                    'inverse_classes_': {idx: label for label, idx in {
                        label: idx for idx, label in enumerate(unique_classes)
                    }.items()},
                    'min_idx': 0,
                    'max_idx': len(unique_classes) - 1
                }
        else:
            raise ValueError("Input must be a pandas Series or DataFrame.")

        self.fitted = True
        return self

    def transform(self, X):
        """
        Transform labels into numeric representation.
        """
        if not self.fitted:
            raise ValueError("CustomLabelEncoder has not been fitted yet.")

        if isinstance(X, pd.Series):
            return X.map(self.encoders[0]['classes_']).fillna(-1).astype(int)
        elif isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for col in self.encoders:
                X_transformed[col] = X_transformed[col].map(self.encoders[col]['classes_']).fillna(-1).astype(int)
            return X_transformed
        else:
            raise ValueError("Input must be a pandas Series or DataFrame.")

    def inverse_transform(self, X):
        """
        Convert numeric representation back to original labels.
        """
        if not self.fitted:
            raise ValueError("CustomLabelEncoder has not been fitted yet.")

        def adjust_index(col, idx):
            if idx < self.encoders[col]['min_idx']:
                return self.encoders[col]['inverse_classes_'][self.encoders[col]['min_idx']]
            elif idx > self.encoders[col]['max_idx']:
                return self.encoders[col]['inverse_classes_'][self.encoders[col]['max_idx']]
            return self.encoders[col]['inverse_classes_'].get(idx, None)

        if isinstance(X, pd.Series):
            return X.apply(lambda x: adjust_index(0, x))
        elif isinstance(X, pd.DataFrame):
            X_inverse = X.copy()
            for col in self.encoders:
                X_inverse[col] = X_inverse[col].apply(lambda x: adjust_index(col, x))
            return X_inverse
        else:
            raise ValueError("Input must be a pandas Series or DataFrame.")

    def fit_transform(self, X):
        """
        Fit and transform the labels in one step while keeping headers.
        """
        self.fit(X)
        return self.transform(X)


# Example usage:
if __name__ == "__main__":
    df = pd.DataFrame({
        'Animal': ['cat', 'dog', 'fish', 'cat', 'dog', 'bird'],
        'Color': ['red', 'blue', 'green', 'blue', 'red', 'green'],
        'Age': [3, 5, 2, 4, 7, 1]  # This should remain unchanged
    })

    selected_columns = ['Animal', 'Color']

    print("Original DataFrame:\n", df)

    encoder = CustomEncoder()
    df[selected_columns] = encoder.fit_transform(df[selected_columns])  # Replace selected columns

    print("\nEncoded DataFrame:\n", df)

    df[selected_columns] = encoder.inverse_transform(df[selected_columns])  # Decode selected columns
    print("\nDecoded DataFrame:\n", df)
