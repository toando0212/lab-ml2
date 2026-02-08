import pypdf
import os

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        return f"File not found: {pdf_path}"
    
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        # Trích xuất 3 trang đầu để nắm bắt Abstract, Introduction và Conclusion
        for i in range(min(3, len(reader.pages))):
            text += f"\n--- Page {i+1} ---\n"
            text += reader.pages[i].extract_text()
        
        # Trích xuất trang cuối để xem Conclusion
        if len(reader.pages) > 3:
            text += f"\n--- Last Page ({len(reader.pages)}) ---\n"
            text += reader.pages[-1].extract_text()
            
        return text
    except Exception as e:
        return f"Error reading {pdf_path}: {str(e)}"

def main():
    paper_dir = "/Volumes/Toan/ML2/related_paper"
    papers = ["ellahyani2016.pdf", "zaklouta2011.pdf"]
    
    for paper in papers:
        full_path = os.path.join(paper_dir, paper)
        print(f"\n==========================================")
        print(f"CONTENT OF: {paper}")
        print(f"==========================================")
        print(extract_text_from_pdf(full_path))

if __name__ == "__main__":
    main()
