import textwrap
from pathlib import Path

import streamlit as st

from backend.email_agent import draft_email_reply
from config import settings
from rag_chain import build_email_rag_chain


def _kb_status() -> str:
    kb_file = Path(settings.kb_path)
    if not kb_file.exists():
        return "Missing"
    if not Path(settings.vectordb_path).exists():
        return "KB file found, index missing"
    return "Ready"


def setup_page() -> None:
    st.set_page_config(
        page_title="Referral Email Assistant",
        page_icon="📧",
        layout="wide",
    )

    # Simple modern styling
    st.markdown(
        """
        <style>
        .main {
            background: radial-gradient(circle at top left, #1f2933 0, #111827 55%, #020617 100%);
            color: #e5e7eb;
        }
        .stTextArea textarea {
            background-color: #020617 !important;
            border-radius: 10px !important;
            border: 1px solid #374151 !important;
            color: #e5e7eb !important;
        }
        .stButton>button {
            border-radius: 999px;
            padding: 0.6rem 1.8rem;
            border: none;
            background: linear-gradient(135deg, #22c55e, #16a34a);
            color: white;
            font-weight: 600;
            box-shadow: 0 10px 25px rgba(34, 197, 94, 0.25);
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #16a34a, #15803d);
        }
        .card {
            background: rgba(15, 23, 42, 0.9);
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(55, 65, 81, 0.8);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.title("📊 Status")
        st.markdown("**Knowledge base**")
        st.write(f"File: `{settings.kb_path}`")
        st.write(f"Index: `{settings.vectordb_path}`")
        st.write(f"Status: {_kb_status()}")

        st.markdown("---")
        st.markdown("**Model**")
        st.write(f"Groq model: `{settings.groq_model_name}`")

        st.markdown("---")
        st.markdown(
            textwrap.dedent(
                """
                **Usage tips**

                - Paste the full client email (subject + body).
                - The reply is generated strictly from the internal KB.
                - If the KB does not cover the query, the fixed escalation
                  message is returned.
                """
            )
        )


def render_main() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            background-color: #f5f5f5;
            color: #333333;
        }

        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border: none;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #45a049;
        }

        .stTextArea>div>textarea {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 8px;
            padding: 12px;
            font-family: 'Inter', sans-serif;
            font-size: 14px;
        }

        .stTextArea>div>textarea:focus {
            border-color: #4CAF50;
            outline: none;
        }

        .stMarkdown {
            font-size: 18px;
            line-height: 1.6;
        }

        .stHeader {
            font-size: 36px;
            font-weight: 700;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 20px;
        }

        .stSubheader {
            font-size: 24px;
            font-weight: 600;
            color: #34495e;
            margin-bottom: 10px;
        }

        .card {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
        }

        .footer {
            text-align: center;
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h1 class='stHeader'>Welcome to Your Email Assistant</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='stMarkdown'>Effortlessly draft professional and compliant email replies with the power of AI. "
        "Streamline your communication and save time with our intuitive interface.</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Client Email")
        email_text = st.text_area(
            "Paste the client's email here:",
            height=200,
            placeholder="Enter the email content...",
        )
        generate = st.button("Generate Reply")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Drafted Reply")
        output_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        if generate:
            if not email_text.strip():
                st.warning("Please paste a client email first.")
                return

            try:
                with st.spinner("Drafting reply from knowledge base..."):
                    user_name = "Raghav Kankane"
                    company_name = "Kankane Prints Stacks Pvt Ltd"
                    chain = build_email_rag_chain(
                        user_name=user_name, company_name=company_name
                    )
                    reply = chain.invoke({"email": email_text})
                output_placeholder.markdown(reply.content)
            except FileNotFoundError:
                st.error(
                    "The vector index could not be found. "
                    "Please run `python ingest.py` once to build it."
                )
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong while generating the reply: {exc}")

    st.markdown(
        "<div class='footer'>Made with ❤️ by Your Team</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    setup_page()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
