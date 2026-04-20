# this was a temporary file to fix an issue in csv formatting.
# import csv

# input_file = "emoji_ai_requirements/broken.csv"
# output_file = "fixed.csv"

# def clean_text(text):
#     # Collapse consecutive quotes to a single quote
#     while '""' in text:
#         text = text.replace('""', '"')
#     return text

# with open(input_file, 'r', encoding='utf-8') as infile, \
#      open(output_file, 'w', newline='', encoding='utf-8') as outfile:

#     writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)

#     header = infile.readline().strip().split(',')
#     writer.writerow(header)

#     for line in infile:
#         line = line.strip()
#         parts = line.rsplit(',', 10)
#         if len(parts) < 11:
#             continue

#         text_col = clean_text(parts[0])
#         other_cols = parts[1:]

#         writer.writerow([text_col] + other_cols)

# print(f"Cleaned CSV saved to {output_file}")
