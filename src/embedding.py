from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import Chroma
import os

def create_embeddings(api_key: str, base_url: str, model_name: str):
    embeddings = UpstageEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
    )
    return embeddings


def build_or_load_vectordb(docs, embeddings, persist_dir, rebuild=False):
    """
    ChromaDB를 새로 만들거나 기존 벡터DB를 로드.
    """
    os.makedirs(persist_dir, exist_ok=True)

    if rebuild:
        if docs is None: # 문서 없을 시 오류 리턴
            raise ValueError("❌ 업로드 된 PDF 문서가 없습니다. 업로드 후 다시 시도해주세요.")
        print("🔄 새 벡터DB 생성 중...")
        db = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        db.persist()
        print("✅ 벡터DB 생성 완료")
        return db
    else:
        print("📂 기존 벡터DB 로드 중...")
        db = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        print("✅ 벡터DB 로드 완료")
        return db
