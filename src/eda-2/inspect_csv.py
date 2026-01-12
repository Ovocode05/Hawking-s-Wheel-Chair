import csv
import sys

file_path = r'c:\Users\HP\GitMakesMeHappy\Hawking-s-Wheel-Chair\Normalized_dataset\recordings\Dard\Amish.csv'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        print(f"Headers: {headers}")
        for i, row in enumerate(reader):
            if i < 5:
                print(f"Row {i}: {row}")
            else:
                break
except Exception as e:
    print(f"Error: {e}")
