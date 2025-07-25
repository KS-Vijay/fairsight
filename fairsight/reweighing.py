import pandas as pd

class Reweighing:
    """
    Reweighing bias mitigation algorithm (Kamiran and Calders, 2012).
    Computes instance weights to balance the distribution of protected groups and labels.
    Usage: Use the computed weights in model training to mitigate bias.
    """
    def __init__(self, protected_attr: str, label: str, privileged_value, unprivileged_value, positive_label=1, negative_label=0):
        self.protected_attr = protected_attr
        self.label = label
        self.privileged_value = privileged_value
        self.unprivileged_value = unprivileged_value
        self.positive_label = positive_label
        self.negative_label = negative_label

    def compute_weights(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute reweighing weights for each instance in the DataFrame.
        Returns a pandas Series of weights aligned with df.index.
        """
        # Marginal probabilities
        total = len(df)
        p_prot = df[self.protected_attr].value_counts(normalize=True).to_dict()
        p_label = df[self.label].value_counts(normalize=True).to_dict()

        # Joint probabilities
        joint = df.groupby([self.protected_attr, self.label]).size() / total

        # Compute weights
        def get_weight(row):
            prot = row[self.protected_attr]
            label = row[self.label]
            # Marginal
            p_a = p_prot.get(prot, 0)
            p_y = p_label.get(label, 0)
            # Joint
            p_ay = joint.get((prot, label), 0)
            if p_ay == 0:
                return 1.0  # Avoid division by zero
            return (p_a * p_y) / p_ay

        weights = df.apply(get_weight, axis=1)
        return weights

# Example usage (not run):
# rw = Reweighing('gender', 'outcome', privileged_value='male', unprivileged_value='female')
# weights = rw.compute_weights(df) 