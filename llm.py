from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from config import answer_examples


index_name = 'tax-index'

store = {} #store 객체 생성

#session id를 통해 store에 메세지를 저장
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#질문 전처리용 체인
def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해 주세요.
        만약 변경할 필요가 없다고 판단 된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해 주세요.
        사전: {dictionary}
        질문: {{question}}                                     
    """)

    dictionary_chain = prompt | llm | StrOutputParser()

    return dictionary_chain


#GPT 모델 설정
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)

    return llm

# 벡터 검색기 설정
def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')# 3072
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return retriever


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    #prompt = hub.pull("rlm/rag-prompt")


    #프롬프트 템플릿
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"), #히스토리에 대한 정보를 여기에 넣음
        ("human", "{input}"),
        ]
    )

    #rangchain에서 retriever 가져옴 (qa_chain->rag_chain)
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    #위 내용은 과거 기록을 만들기 위한 것
    #create_history_aware_retriever가 context를 책임짐

    return history_aware_retriever


# RAG 체인 구성
def get_rag_chain():
    
    #아래 내용은 질문에 대한 답을 만들기 위한 것

    llm = get_llm()

    # This is a prompt template used to format each individual example.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    history_aware_retriever = get_history_retriever()

    system_prompt = (
    "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요."
    "아래에 제공된 문서를 활용해서 답변해주시고, 답변을 알 수 없다면 모른다고 답변해주세요."
    "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
    "2~3 문장 정도의 짧은 내용의 답변을 원합니다."
    "\n\n"
    "{context}"
    )

    #qa 체인 진행할 때 이를 기반으로 진행
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt), #과거기록을 바탕으로 system prompt 동작
            few_shot_prompt, #퓨샷도 넣음
            MessagesPlaceholder("chat_history"), #과거기록 정보
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    #하나의 문서로 만드는 역할

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    #retrieval로 ragchain 연결(history와 question_answer)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history, #여기에 히스토리 전달
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer') #answer만 보고 싶다

    return conversational_rag_chain

#메인 실행 함수
def get_ai_message(user_message): 
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain() #이거 리턴값 conversational_rag_chain
    
    tax_chain = {"input": dictionary_chain} | rag_chain
    ai_message = tax_chain.stream( #스트림 출력
        {"question": user_message},
        config={"configurable": {"session_id": "abc123"}} #세션 정보
    )        
    
    return ai_message