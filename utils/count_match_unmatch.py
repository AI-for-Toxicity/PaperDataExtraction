import sys
import re

def calculate_totals(file_path):
    total_matched = 0
    total_unmatched = 0
    lines_read = 0
    
    # \s* allows for optional spaces, re.IGNORECASE handles 'Matched' or 'MATCHED'
    matched_pattern = re.compile(r"matched\s*=\s*(\d+)", re.IGNORECASE)
    unmatched_pattern = re.compile(r"unmatched\s*=\s*(\d+)", re.IGNORECASE)
    
    try:
        # Try UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                lines_read += 1
                matched = matched_pattern.search(line)
                unmatched = unmatched_pattern.search(line)
                
                if matched and unmatched:
                    total_matched += int(matched.group(1))
                    total_unmatched += int(unmatched.group(1))
                    
        print(f"[Debug] Read {lines_read} lines from file.")
        print(f"Total matched:   {total_matched}")
        print(f"Total unmatched: {total_unmatched}")
        
    except UnicodeDecodeError:
        print("\n[!] Encoding error detected.")
        print("This file was likely created by PowerShell (UTF-16).")
        print("Change `encoding='utf-8'` to `encoding='utf-16'` on line 15 of this script and try again.\n")
    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <path_to_your_file.txt>")
    else:
        calculate_totals(sys.argv[1])