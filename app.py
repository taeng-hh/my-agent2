
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



from decimal import Decimal, ROUND_HALF_UP
from langchain.tools import tool
import requests
import os


@tool
def get_usd2krw_rate(amount: float = 1.0) -> str:
    """
    Frankfurter API로 최신 USD/KRW 환율을 조회하고
    입력받은 달러 금액을 원화로 환산합니다.

    Args:
        amount: 원화로 환산할 미국 달러 금액.
                금액이 없으면 기본값은 1달러입니다.
    """

    url = os.getenv("FRANKFURTER_API_URL")

    if not url:
        return "환율 API URL이 설정되지 않았습니다."

    if not 0 <= amount <= 1e12:
        return "올바른 달러 금액을 입력해 주세요."

    try:
        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        # 실제 환율 및 기준일
        rate = Decimal(str(data["rate"]))
        rate_date = data["date"]

        # 달러 금액 × 환율
        converted = Decimal(str(amount)) * rate

        # 원 단위 반올림
        converted = converted.quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP
        )

        return (
            f"환율 기준일: {rate_date}\n"
            f"1달러 = {rate:,.2f}원\n"
            f"{amount:g}달러 = {converted:,.0f}원\n"
            f"출처: Frankfurter Exchange Rates API"
        )

    except requests.exceptions.RequestException as e:
        return f"환율 API 요청 실패: {e}"

    except (KeyError, ValueError, TypeError) as e:
        return f"환율 데이터 처리 실패: {e}"

agent = create_agent(
    model=model,
    tools=[get_usd2krw_rate],
    system_prompt="""
    당신은 USD/KRW 환율 조회 및 환전 계산 챗봇입니다.

    규칙:
    1. 달러 환율 또는 달러를 원화로 환산하는 질문에는
       반드시 get_usd2krw_rate 도구를 사용하세요.

    2. 사용자가 금액을 입력하면 amount 인자로 전달하세요.
       예: "100달러는 몇 원?" -> amount=100

    3. 금액 없이 환율만 질문하면 amount=1로 호출하세요.

    4. 도구가 반환한 실제 환율, 환전 결과, 기준일을
       모두 사용하세요.

    5. 환율이나 환전 금액을 임의로 생성하지 마세요.

    6. 환율 조회와 USD/KRW 환전 계산만 지원합니다.

    7. API 호출에 실패하면 실패 사실만 알려주세요.
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
