from pypdf import PdfReader

reader = PdfReader("data/iphone_user_guide.pdf")

for i in range(29, 35):
    print(f"\n--- PAGE {i+1} ---")
    print(reader.pages[i].extract_text())