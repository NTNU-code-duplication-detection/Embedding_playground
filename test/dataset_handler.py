from sqlalchemy import create_engine
import pandas as pd
import os
import getpass

user = getpass.getuser()

# --- Setup ---
engine = create_engine(
    f"postgresql+psycopg2://{user}@localhost:5432/bigclonebench"
)
output_dir = "type1_clones_real_code"
os.makedirs(output_dir, exist_ok=True)

# --- Step 1: Get Type-1 clone pairs ---
clone_df = pd.read_sql("""
    SELECT function_id_one, function_id_two
    FROM clones
    WHERE syntactic_type = 2
    LIMIT 50
""", engine)  # test small first

print(f"Found {len(clone_df)} Type-1 clone pairs")

# --- Step 2: Get real code for all involved function IDs ---
function_ids = set(clone_df['function_id_one']).union(clone_df['function_id_two'])
ids_str = ','.join(map(str, function_ids))

code_df = pd.read_sql(f"""
    SELECT function_id, text
    FROM pretty_printed_functions
    WHERE function_id IN ({ids_str})
""", engine)

# Convert to dict for fast lookup
id_to_code = dict(zip(code_df['function_id'], code_df['text']))

# --- Step 3: Save each clone pair to separate files ---
for idx, row in clone_df.iterrows():
    f1_id = row['function_id_one']
    f2_id = row['function_id_two']

    f1_code = id_to_code.get(f1_id, "")
    f2_code = id_to_code.get(f2_id, "")

    # For Type-2, we don't require exact equality
    pair_dir = os.path.join(output_dir, f"pair_{f1_id}_{f2_id}")
    os.makedirs(pair_dir, exist_ok=True)

    with open(os.path.join(pair_dir, f"{f1_id}.java"), 'w') as f:
        f.write(f1_code)
    with open(os.path.join(pair_dir, f"{f2_id}.java"), 'w') as f:
        f.write(f2_code)


print("Saved all Type-1 clone pairs with original code (no X renaming).")
