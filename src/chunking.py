import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document

def load_and_chunk_pdfs(pdf_dir: str, chunk_size: int = 250, chunk_overlap: int = 50) -> List[Document]:
    """
    PDF 파일을 로드하고 텍스트를 청크 단위로 분할.

    Args:
        pdf_dir (str): PDF 폴더 경로
        chunk_size (int): 각 청크의 최대 길이 (문자 수)
        chunk_overlap (int): 청크 간 중복 길이

    Returns:
        list[Document]: 분할된 Document 리스트
    """
    all_documents = [] 

    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"📂 PDF 폴더가 없습니다: {pdf_dir}")

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError("❌ PDF 파일이 없습니다. 업로드 후 다시 시도하세요.")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(pdf_dir, pdf_file))
        print(f"📄 {pdf_file} 로드 중...")
        all_documents.extend(loader.load()) 


    financial_separators = [
        "\n\n제\s*\d+\s*[조|장|관]\s*\(.+\)", # 예: "제 1 조 (목적)" 형태
        "\n\n제\s*\d+\s*[조|장|관]",          # 예: "제 1 조" 형태
        "\n\n",                             # 일반적인 단락 구분자
        "\n",                               # 줄 바꿈
        " ",                                # 공백
        "",
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=financial_separators, 
    )
    
   
    all_chunks = splitter.split_documents(all_documents)
    
    print(f"📘 총 {len(all_chunks)}개 청크 생성 완료")
    return all_chunks