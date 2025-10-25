import json
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_upstage import ChatUpstage
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# 💡 LLM 기반 점수 평가 함수 (MMR에 사용할 점수 생성)
def _get_llm_relevance_scores(api_key: str, base_url: str, chat_model: str, query: str, documents: List[Document]) -> List[float]:
    """LLM을 사용하여 각 문서의 관련성 점수를 0~100 사이로 평가합니다."""
    llm = ChatUpstage(
        api_key=api_key,
        base_url=base_url,
        model=chat_model,
        temperature=0.0, # 일관성을 위해 0.0 설정
    )
    
    # JSON 출력을 유도하는 프롬프트
    template = """
    너는 문서 관련성을 평가하는 전문가야. 아래의 사용자 질문과 문서 내용을 보고, 
    질문에 답변하는 데 이 문서가 얼마나 유용한지 0점부터 100점 사이의 점수로 평가해.

    ---
    사용자 질문: {query}

    ---
    문서 내용:
    {document_content}

    ---
    반드시 다음 JSON 형식으로만 응답해야 해. 다른 설명이나 텍스트는 일절 추가하지 마.
    {{"relevance_score": 점수}}
    """
    
    prompt = PromptTemplate.from_template(template)
    scores = []
    
    #모든 문서에 대해 LLM 평가 실행
    for doc in documents:
        try:
            # LLM Chain 실행
            response = llm.invoke(
                prompt.format(
                    query=query, 
                    document_content=doc.page_content
                )
            ).content
            
            # 🟢 JSON 추출 강화: 응답에서 첫 번째 JSON 객체를 정규 표현식으로 추출
            match = re.search(r'\{.*\}', response.strip(), re.DOTALL)
            
            if match:
                score_json_str = match.group(0)
                score_json = json.loads(score_json_str)
                score = float(score_json.get("relevance_score", 0)) / 100.0  # 0~1 범위로 정규화
                scores.append(score)
                doc.metadata['llm_relevance_score'] = score * 100.0 # 메타데이터에는 0~100 저장
            else:
                raise ValueError("LLM 응답에서 유효한 JSON 객체를 찾을 수 없습니다.")

        except Exception as e:
            # 파싱 오류 또는 기타 오류 발생 시 0점으로 처리
            print(f"⚠️ LLM 점수 평가 중 오류 발생: {e}")
            scores.append(0.0)
            doc.metadata['llm_relevance_score'] = 0.0
            
    return scores

# MMR 알고리즘을 사용하여 최종 문서 순위 결정
def rerank_documents_mmr(api_key: str, base_url: str, chat_model: str, embeddings: object, query: str, documents: List[Document], top_k: int = 5, lambda_mult: float = 0.5) -> List[Document]:
    """
    LLM 관련성 점수를 기준으로 MMR(Maximal Marginal Relevance)을 적용하여 문서를 재순위화합니다.

    Args:
        api_key, base_url, chat_model: LLM 점수 평가용 인자.
        embeddings (object): 임베딩 모델 객체 (src/embedding.py에서 가져온 것).
        query (str): 사용자 질문.
        documents (List[Document]): Retriever로부터 검색된 문서 리스트.
        top_k (int): 최종적으로 선택할 문서의 개수.
        lambda_mult (float): MMR 람다 값 (0: 다양성 우선, 1: 관련성 우선).

    Returns:
        List[Document]: MMR을 통해 재순위화된 상위 K개 Document 객체 리스트.
    """
    if not documents or top_k == 0:
        return []

    # 1. LLM을 사용하여 문서 관련성 점수 획득 (Relevance Score)
    llm_scores = _get_llm_relevance_scores(api_key, base_url, chat_model, query, documents)
    
    # 2. 문서 텍스트 임베딩 생성 (Diversity Score 계산용)
    document_texts = [doc.page_content for doc in documents]
    
    # 주의: embeddings 객체는 langchain의 UpstageEmbeddings 객체여야 합니다.
    try:
        doc_embeddings = np.array(embeddings.embed_documents(document_texts))
    except Exception as e:
        print(f"❌ 임베딩 생성 실패: {e}. 'embeddings' 객체가 올바른 LangChain Embeddings 객체인지 확인하세요.")
        return []

    # 3. 문서 간의 코사인 유사도 행렬 계산 (Diversity Term)
    # 문서가 1개일 경우를 대비하여 np.eye(N)으로 초기화
    if len(doc_embeddings) == 1:
        similarity_matrix = np.array([[1.0]])
    else:
        similarity_matrix = cosine_similarity(doc_embeddings)

    # 4. MMR 알고리즘 적용
    
    # 초기화
    N = len(documents)
    selected_indices = []
    unselected_indices = list(range(N))
    
    for _ in range(min(top_k, N)):
        if not unselected_indices:
            break

        mmr_scores = []
        
        # 선택되지 않은 각 문서에 대해 MMR 점수 계산
        for i in unselected_indices:
            # 관련성 항 (Relevance Term): LLM 점수 (0~1)
            relevance_term = llm_scores[i]
            
            # 다양성 항 (Diversity Term): 선택된 문서들과의 최대 유사도 (낮을수록 좋음)
            if not selected_indices:
                diversity_term = 0.0
            else:
                max_similarity = np.max(similarity_matrix[i, selected_indices])
                diversity_term = max_similarity
            
            # MMR = lambda * Relevance - (1 - lambda) * Diversity
            mmr_score = (lambda_mult * relevance_term) - ((1 - lambda_mult) * diversity_term)
            mmr_scores.append((mmr_score, i))
        
        # MMR 점수가 가장 높은 문서 선택
        best_score, best_index = max(mmr_scores)
        
        # 선택된 인덱스 추가 및 미선택 인덱스에서 제거
        selected_indices.append(best_index)
        unselected_indices.remove(best_index)

    # 5. 최종 문서 리스트 생성
    final_reranked_docs = [documents[i] for i in selected_indices]
    
    return final_reranked_docs

# app.py에서 호출할 최종 함수명
rerank_documents = rerank_documents_mmr