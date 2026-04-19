from neuroCombat import neuroCombat
import pandas as pd

def apply_neurocombat(features, sites):
    data = features.T

    covars = pd.DataFrame({
        "site": sites
    })

    result = neuroCombat(
        dat=data,
        covars=covars,
        batch_col="site"
    )

    return result["data"].T