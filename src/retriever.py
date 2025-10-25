from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_upstage import ChatUpstage

def build_retrieval_qa(api_key, base_url, model_name, vectordb, top_k=5):
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    prompt = PromptTemplate(
        input_variables=["context", "question"], 
        template=(
        """당신은 금융상품 약관을 설명하는 전문가입니다.
        주어진 문맥을 바탕으로 질문에 대해 정확하고 이해하기 쉽게 답변해주세요.
        답변할 때는 다음 사항을 지켜주세요:
        1. 약관 내용을 기반으로만 답변
        2. 전문 용어는 쉽게 설명
        3. 중요한 내용은 강조
        4. 답변을 찾을 수 없으면 "약관에서 해당 내용을 찾을 수 없습니다"라고 답변
        문맥: {context}
        질문: {question}
        답변:"""
        ),
    )

    llm = ChatUpstage(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.2,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff", # 문서들을 하나의 프롬프트로 묶어 LLM에 전달하는 방식
        return_source_documents=True, # 답변에 사용된 소스 문서 반환 설정
        chain_type_kwargs={"prompt": prompt},
        input_key="question",
        output_key="result"   
    )

    return qa_chain
