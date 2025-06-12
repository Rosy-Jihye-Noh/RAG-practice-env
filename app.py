import streamlit as st
import os

from dotenv import load_dotenv

from llm import get_ai_message


st.set_page_config(
    page_title="소득세 챗봇",
    page_icon=":guardsman:",  # 이모지 아이콘
    #layout="wide",  # 레이아웃 설정
    #initial_sidebar_state="expanded"  # 사이드바 초기 상태
)

st.title("Stream 기본 예제")
st.caption("소득세에 관련된 모든것을 답변해 드립니다.")

load_dotenv()

# pinecone_api_key = os.getenv('PINECONE_API_KEY')
# pc = Pinecone(api_key=pinecone_api_key)
#최초 1번 로드 하면 됨.

if "message_list" not in st.session_state:
    st.session_state.message_list = []
    #스트림릿은 실행할 때마다 리프레시됨. 그래서 세션 스테이트를 제공하여 message_list에 추가가


for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # role과 content 연결되어 출력력
                

if user_question := st.chat_input(placeholder="소득세에 관련 궁금한 내용을 말씀해 주세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})
    
    
    with st.spinner("답변을 생성하는 중입니다."):
        ai_response = get_ai_message(user_question)    
        with st.chat_message('ai'):
           ai_message = st.write_stream(ai_response)
           # 이렇게 앞에 ai_message = 를 적어야 기존 답변이 없어지지 않음
        st.session_state.message_list.append({"role": "ai", "content": ai_message})        
        
      