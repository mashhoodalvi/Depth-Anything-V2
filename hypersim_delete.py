import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--decompress_dir", default="./Hypersim")
parser.add_argument("--dry_run", action="store_true", help="Don't actually delete, just print what would be done")
args = parser.parse_args()

def delete_recursively(root_dir: str, substrings: list, dry_run: bool = True):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            file_path = os.path.join(dirpath, fname)
            # match against filename (not full path) to avoid accidental matches
            if any(s in fname for s in substrings):
                print(f"Skipping (matched): {file_path}")
            else:
                if dry_run:
                    print(f"[DRY RUN] Would delete: {file_path}")
                else:
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    print("Top-level entries:", os.listdir(args.decompress_dir))
    ans = input("Delete files in those directories? type [y] to delete or any other key to cancel: ")
    if ans.lower() == "y":
        delete_recursively(args.decompress_dir, ["tonemap", "depth_meters", "semantic"], dry_run=args.dry_run)
    else:
        print("Cancelled.")



