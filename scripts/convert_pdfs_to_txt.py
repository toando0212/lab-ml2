import pypdf
import os

def convert_pdf_to_txt(pdf_path, txt_path):
    print(f"Processing: {pdf_path}...")
    try:
        reader = pypdf.PdfReader(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                f.write(f"\n--- PAGE {i+1} ---\n")
                f.write(page.extract_text())
        print(f"✅ Saved to: {txt_path}")
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")

def main():
    paper_dir = "/Volumes/Toan/ML2/related_paper"
    output_dir = "/Volumes/Toan/ML2/related_paper" # Lưu cùng thư mục cho dễ quản lý
    
    for filename in os.listdir(paper_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(paper_dir, filename)
            txt_path = os.path.join(output_dir, filename.replace(".pdf", ".txt"))
            convert_pdf_to_txt(pdf_path, txt_path)

if __name__ == "__main__":
    main()
