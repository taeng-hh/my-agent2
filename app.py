
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
import requests


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FRANKFURTER_API_URL = os.getenv("FRANKFURTER_API_URL")

# 환경변수 확인
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY가 없습니다.")
    st.stop()

if not FRANKFURTER_API_URL:
    st.error("FRANKFURTER_API_URL이 없습니다.")
    st.stop()


st.set_page_config(
    page_title="실시간 환율 조회 챗봇",
    page_icon="💱"
)

st.title("💱 실시간 환율 조회 챗봇")

st.caption(
    "Frankfurter API 기반 달러/원 최신 환율 조회 "
    "(일별 갱신)"
)


model = init_chat_model(
    "gpt-5.4-mini",
    model_provider="openai"
)


@tool
def get_usd2krw_rate() -> str:
    """Frankfurter API를 사용하여 USD/KRW 최신 환율을 조회합니다."""

    url = os.getenv("FRANKFURTER_API_URL")

    if not url:
        return "FRANKFURTER_API_URL 환경변수가 없습니다."

    try:
        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        rate = float(data["rate"])
        rate_date = data["date"]

        return (
            f"{rate_date} 기준 "
            f"1달러당 {rate:,.2f}원입니다."
        )

    except requests.exceptions.RequestException as e:
        return f"환율 API 요청 실패: {e}"

    except (KeyError, ValueError, TypeError) as e:
        return f"환율 데이터 처리 실패: {e}"

agent = create_agent(
    model=model,
    tools=[get_usd2krw_rate],
    system_prompt="""
    당신은 USD/KRW 최신 환율 조회 전용 챗봇입니다.

    규칙:
    1. 사용자가 달러/원 환율을 질문하면
       반드시 get_usd2krw_rate 도구를 호출하세요.

    2. 도구가 반환한 실제 환율과 기준일만 사용하세요.

    3. 환율을 직접 추측하거나 만들어내지 마세요.

    4. 웹 검색을 사용하지 마세요.

    5. 다른 통화 환율, 환전 계산,
       과거 환율 등은 지원하지 않습니다.

    6. 환율 조회 이외의 질문에는
       "USD/KRW 최신 환율 조회만 지원합니다."
       라고 답변하세요.

    7. API 호출에 실패하면
       환율을 임의로 생성하지 말고
       조회 실패 사실을 안내하세요.
    """
)


if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------------
# 7. 이전 대화 출력
# -----------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


prompt = st.chat_input("여기에 입력하시오")

if prompt:

    # 사용자 질문 저장
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        with st.spinner("최신 환율 조회 중..."):

            try:
                response = agent.invoke({
                    "messages": st.session_state.messages
                })

                answer = response["messages"][-1].content

            except Exception as e:
                answer = (
                    "환율 조회 중 오류가 발생했습니다. "
                    f"오류 유형: {type(e).__name__}"
                )

        st.write(answer)

    # AI 답변 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
