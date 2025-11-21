import csv
import zipfile
import os
import argparse
from pathlib import Path

def extract_small_csv(zip_path: str, inner_csv_name: str, out_csv: str, limit: int = 50, start: int = 0):
    """Extract the first `limit` rows (including header) from a CSV file inside a ZIP archive.

    Arguments:
    - zip_path: path to the zip archive
    - inner_csv_name: the exact filename within the zip (like 'job_descriptions.csv')
    - out_csv: where to write the smaller CSV
    - limit: number of data rows to include (not counting header)
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        namelist = z.namelist()
        if inner_csv_name not in namelist:
            raise FileNotFoundError(f"{inner_csv_name} not found in zip archive. Available: {namelist[:10]}...")

        with z.open(inner_csv_name, 'r') as csvfile_in, open(out_csv, 'w', newline='', encoding='utf-8') as csvfile_out:
            # csvfile_in is a binary file; wrap it
            reader = csv.reader(line.decode('utf-8', errors='ignore') for line in csvfile_in)
            writer = csv.writer(csvfile_out)

            # Write header
            try:
                header = next(reader)
            except StopIteration:
                # Empty file
                return
            writer.writerow(header)

            # Skip `start` rows (data rows, header already consumed)
            if start < 0:
                raise ValueError("start must be >= 0")
            if limit < 0:
                raise ValueError("limit must be >= 0")

            skipped = 0
            if start > 0:
                for _ in range(start):
                    try:
                        next(reader)
                        skipped += 1
                    except StopIteration:
                        # file exhausted before we could skip
                        break
            if skipped < start:
                print(f"Warning: requested start={start} but file only had {skipped} data rows to skip. Output will start from EOF if any.")

            # Write up to `limit` rows
            count = 0
            for row in reader:
                writer.writerow(row)
                count += 1
                if count >= limit:
                    break

            print(f"Wrote {count} rows (plus header) to {out_csv}")
            if count == 0:
                print("Note: No rows written; check that `start` and `limit` values are in the CSV's range.")


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    parser = argparse.ArgumentParser(description='Extract a slice from a large CSV inside a ZIP archive.')
    parser.add_argument('--zip', '-z', type=str, default=str(data_dir / 'job-description-dataset.zip'), help='Path to source zip file')
    parser.add_argument('--inner', '-i', type=str, default='job_descriptions.csv', help='Name of CSV inside zip')
    parser.add_argument('--out', '-o', type=str, default=str(data_dir / 'job_descriptions_small.csv'), help='Where to write smaller CSV')
    parser.add_argument('--limit', '-l', type=int, default=50, help='Number of rows to extract (data rows, not counting header)')
    parser.add_argument('--start', '-s', type=int, default=0, help='Starting data-row index (0-based, header is not counted)')

    args = parser.parse_args()
    extract_small_csv(args.zip, args.inner, args.out, limit=args.limit, start=args.start)
