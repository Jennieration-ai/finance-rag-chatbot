# app.py
import os
import time
import streamlit as st
from dotenv import load_dotenv

from src.chunking import load_and_chunk_pdfs
from src.embedding import create_embeddings, build_or_load_vectordb
from src.retriever import build_retrieval_qa
from src.reranker import rerank_documents  

# ─────────────────────────────────────────────────────────────
# 환경 변수 (.env) 로드 
# ─────────────────────────────────────────────────────────────
load_dotenv()

API_KEY     = os.getenv("UPSTAGE_API_KEY", "")
BASE_URL    = os.getenv("UPSTAGE_API_BASE_URL", "https://api.upstage.ai/v1")
CHAT_MODEL  = os.getenv("UPSTAGE_CHAT_MODEL", "solar-pro-2")
EMBED_MODEL = os.getenv("UPSTAGE_EMBED_MODEL", "solar-embedding-1-large-passage")

# ─────────────────────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="금융상품 약관 AI 도우미", page_icon="🏦", layout="wide")

PDF_DIR      = "./data/pdfs"
VECTORDB_DIR = "./vectordb"
os.makedirs(PDF_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────────────────────────────────
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None  
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role:"user"/"assistant", content:str, sources:[...]}]
if "last_index_time" not in st.session_state:
    st.session_state.last_index_time = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ─────────────────────────────────────────────────────────────
# 스타일 (간단 테마)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 헤더 배너 */
.app-hero {
  padding: 24px 28px; border-radius: 16px;
  background: linear-gradient(135deg, #EEF4FF, #F7FAFD);
  border: 1px solid #E5ECFF; margin-bottom: 14px;
}
.example-chip {
  padding: 10px 14px; border-radius: 999px;
  border: 1px solid #e6e6e6; background: #fff; font-size: 14px;
  cursor: pointer; display: inline-block; margin-right: 8px; margin-bottom:8px;
}
.side-note { font-size: 12px; color: #6b7280; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 사이드바 – 업로드/파일목록/색인 컨트롤/히스토리 관리
# ─────────────────────────────────────────────────────────────
pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])

@st.dialog("📂 업로드된 파일 목록")
def show_uploaded_files():
    if pdf_files:
        st.write("현재 저장된 PDF 파일은 다음과 같습니다 👇")
        for i, f in enumerate(pdf_files, start=1):
            size_kb = os.path.getsize(os.path.join(PDF_DIR, f)) / 1024
            st.markdown(f"**{i}. {f}** ({size_kb:.0f} KB)")
    else:
        st.info("아직 업로드된 PDF가 없습니다.")
    st.markdown("---")
    st.caption("📘 새 문서를 추가하려면 PDF 업로드 기능을 이용하세요.")

with st.sidebar:
    st.header("📚 문서 관리")
    uploaded_files = st.file_uploader(
        "PDF 업로드", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(PDF_DIR, file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success(f"{len(uploaded_files)}개의 PDF 파일이 업로드되었습니다 ✅")

    if st.button("📄 업로드된 파일 보기"):
        show_uploaded_files()

    # st.markdown("---")

    # 수동 색인
    if st.button("📂 문서 불러와서 분석하기"):
        if not pdf_files:
            st.error("📂 PDF 문서가 없습니다. 업로드 후 시도하세요.")
        else:
            try:
                with st.spinner("PDF 로드 및 임베딩 중..."):
                    docs = load_and_chunk_pdfs(PDF_DIR)
                    embeddings = create_embeddings(API_KEY, BASE_URL, EMBED_MODEL)
                    vectordb = build_or_load_vectordb(docs, embeddings, VECTORDB_DIR, rebuild=True)
                    st.session_state.embeddings = embeddings
                    st.session_state.vectordb = vectordb
                    st.session_state.last_index_time = time.strftime("%Y-%m-%d %H:%M:%S")
                st.success("✅ 데이터베이스 구축 완료!")
            except Exception as e:
                st.error(f"색인 중 오류: {e}")

    # 기존 색인 자동 로드 (앱 시작 시 1회)
    if os.path.exists(VECTORDB_DIR) and st.session_state.vectordb is None:
        try:
            with st.spinner("기존 색인을 자동으로 불러오는 중..."):
                embeddings = create_embeddings(API_KEY, BASE_URL, EMBED_MODEL)
                vectordb = build_or_load_vectordb(None, embeddings, VECTORDB_DIR, rebuild=False)
                st.session_state.embeddings = embeddings
                st.session_state.vectordb = vectordb
                st.session_state.last_index_time = time.strftime("%Y-%m-%d %H:%M:%S")
            st.success("✅ 기존 데이터베이스가 자동으로 로드되었습니다.")
        except Exception as e:
            st.error(f"기존 색인 로드 오류: {e}")

    # 히스토리 컨트롤
    st.markdown("---")
    st.header("🕘 대화 히스토리")
    if st.session_state.messages:
        for m in st.session_state.messages[::-1][:15]:
            who = "🧑🏻‍💻" if m["role"] == "user" else "🤖"
            st.write(f"{who} {m['content'][:40]}{'…' if len(m['content'])>40 else ''}")
    else:
        st.info("대화가 없습니다.")

    # col1, col2 = st.columns(2)
    # with col1:
    if st.button("🧹 히스토리 초기화"):
        # 대화 기록 및 관련 상태값 초기화
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.session_state.query = None
        st.session_state.response = None
        st.success("💬 모든 대화 기록이 초기화되었습니다!")  
        st.rerun()

    # with col2:
    if st.button("🌀 캐시 정리하기"):
        st.session_state.vectordb = None
        st.session_state.embeddings = None
        st.info("메모리의 DB 핸들을 초기화했습니다.")

# ─────────────────────────────────────────────────────────────
# 메인 – 헤더 + 예시 질문 + 채팅
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="app-hero">
  <h2>🏦 금융상품 약관 AI 도우미</h2>
  <p class="side-note">환영합니다! 사이드바에서 약관 문서를 업로드하고 데이터베이스를 구축해주세요.</p>
</div>
""",
    unsafe_allow_html=True,
)

# 예시 질문 칩
st.write("**💡 이런 질문을 해볼 수 있어요:**")
examples = [
    "대출 시 자금 용도는 제한이 있나요?",
    "담보가치가 떨어지면 어떤 조치를 해야 하나요?",
    "금리가 오르면 나도 이자를 더 내야 하나요?",
    "이자율이 바뀌면 계약을 해지할 수 있나요?",
    "은행이 제 예금을 자동으로 상계할 수 있나요?",
    "대출 계약 철회 시 수수료는 돌려받을 수 있나요?"
]
for i in range(0, len(examples), 3):
    cols = st.columns(3, gap="medium")  # ✅ 'gap="medium"'으로 간격 확보
    for j, col in enumerate(cols):
        if i + j < len(examples):
            q = examples[i + j]
            if col.button(f"💬 {q}", use_container_width=True, key=f"ex_{i+j}"):
                st.session_state.pending_query = q

# 마지막 색인 시각
if st.session_state.last_index_time:
    st.caption(f"마지막 색인 시각: {st.session_state.last_index_time}")

st.markdown("---")
st.subheader("💬 약관에 대해 질문해보세요.")

# 이전전 대화 렌더링 기능은 향후 세션 저장 기능 추가 시 재활성화 예정
# for msg in st.session_state.messages:
#     with st.chat_message("user" if msg["role"] == "user" else "assistant"):
#         st.write(msg["content"])
#         if msg["role"] == "assistant" and msg.get("sources"):
#             with st.expander("🔍 참고 문서"):
#                 for i, d in enumerate(msg["sources"], start=1):
#                     st.markdown(f"**[{i}] {d['source']}**, page {d['page']} (LLM 점수: {d.get('score','-')})")
#                     st.caption(d["preview"])

# 입력
user_input = st.chat_input("예) 중도상환 수수료 면제 조건은?")
if st.session_state.pending_query and not user_input:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

if user_input:
    # 사용자 메시지 기록
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})
    with st.chat_message("user"):
        st.write(user_input)

    vectordb   = st.session_state.vectordb
    embeddings = st.session_state.embeddings

    if not vectordb or not embeddings:
        with st.chat_message("assistant"):
            st.error("벡터DB가 준비되지 않았습니다. 사이드바에서 (재)색인하기를 먼저 실행해주세요.")
    else:
        with st.chat_message("assistant"):
            try:
                # 1) 초기 검색
                top_k_initial = 10
                retriever = vectordb.as_retriever(search_kwargs={"k": top_k_initial})
                with st.spinner(f"📄 관련 문서 {top_k_initial}개 검색 중..."):
                    initial_docs = retriever.invoke(user_input)

                # 2) 리랭킹 (LLM 점수 + MMR)
                top_k_final = 5
                lambda_mult = 0.5
                with st.spinner("🔁 LLM 점수 + MMR 재정렬 중..."):
                    reranked_docs = rerank_documents(
                        api_key=API_KEY,
                        base_url=BASE_URL,
                        chat_model=CHAT_MODEL,
                        embeddings=embeddings,
                        query=user_input,
                        documents=initial_docs,
                        top_k=top_k_final,
                        lambda_mult=lambda_mult
                    )

                # 3) 답변 생성
                chain = build_retrieval_qa(API_KEY, BASE_URL, CHAT_MODEL, vectordb, top_k=top_k_final)
                with st.spinner("💬 답변 생성 중..."):
                    result = chain.invoke({"question": user_input})
                    answer = result["result"]

                # 4) 출력
                st.write(answer)

                # 참고 문서 표시 + 히스토리에 저장
                source_cards = []
                if reranked_docs:
                    with st.expander("🔍 참고 문서"):
                        for i, d in enumerate(reranked_docs, start=1):
                            src = d.metadata.get("source", "(unknown)").split("/")[-1]
                            page = d.metadata.get("page", "?")
                            score = d.metadata.get("llm_relevance_score")
                            score_str = f"{score:.2f}/100" if isinstance(score, (int, float)) else "-"
                            st.markdown(f"**[{i}] {src}**, page {page} (LLM 점수: `{score_str}`)")
                            st.caption(d.page_content[:500] + "...")
                            source_cards.append({
                                "source": src, "page": page, "score": score_str,
                                "preview": (d.page_content[:200] + "...")
                            })

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_cards
                })

            except Exception as e:
                st.error(f"오류가 발생했습니다: {type(e).__name__}: {e}")
