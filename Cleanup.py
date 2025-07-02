import os
import sqlite3

folder = "examples/Heisenberg_Chain/trained_esns"

for filename in os.listdir(folder):
    if filename.endswith(".db"):
        db_path = os.path.join(folder, filename)
        print(f"\n🔍 Inspecting {filename}...")

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get the current study name
            cursor.execute("SELECT study_name FROM studies")
            row = cursor.fetchone()
            if not row:
                print("  ⚠ No study found.")
                conn.close()
                continue

            original_name = row[0]
            print(f"  📌 Found study: {original_name}")

            if "dt" not in original_name:
                print("  ❌ No 'dt' in study name. Skipping.")
                conn.close()
                continue

            # Parse and round the dt
            pre_dt, post_dt = original_name.split("dt", 1)
            dt_str, suffix = post_dt.split("_dpth", 1)
            dt_val = float(dt_str)
            dt_rounded = str(round(dt_val, 5))
            new_name = f"{pre_dt}dt{dt_rounded}_dpth{suffix}"

            if original_name == new_name:
                print("  ✅ Already rounded.")
                conn.close()
                continue

            # Perform SQL update
            print(f"  ✏ Renaming: {original_name} → {new_name}")
            cursor.execute("UPDATE studies SET study_name = ? WHERE study_name = ?", (new_name, original_name))
            conn.commit()
            print("  ✅ Rename successful.")

            conn.close()

        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
