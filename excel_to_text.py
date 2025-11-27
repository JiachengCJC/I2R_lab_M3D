import pandas as pd
import numpy as np

def df_to_custom_string(df):
    # Forward fill main character column
    df_filled = df.copy()
    main_col = df.columns[0]
    df_filled[main_col] = df_filled[main_col].ffill()

    result = []

    # Group by the filled main character
    for main, group in df_filled.groupby(main_col):
        entries = []

        # For each row inside the group
        for _, row in group.iterrows():
            d = {}
            for col in df.columns[1:]:  # skip first column
                d[col] = row[col]
            # Convert dict to the string format {col: value, col2:value}
            entry = "{" + ", ".join([f"{k}:{v}" for k, v in d.items()]) + "}"
            entries.append(entry)

        # join entries with comma
        entry_str = ", ".join(entries)
        
        # format: main: {dict}, {dict};
        result.append(f"{main}: {entry_str};")

    # join all main categories with newline (or anything you prefer)
    return "\n".join(result)
