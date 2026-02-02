from config import settings
from rag_chain import build_email_rag_chain


def main():
    if not settings.is_config_valid:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Please configure your .env file."
        )

    # Provide user_name and company_name here
    user_name = "Raghav Kankane"
    company_name = "mystockbrokers Pvt Ltd"

    chain = build_email_rag_chain(user_name=user_name, company_name=company_name)

    print(
        "Email RAG agent ready. Paste the client's email below, then press Enter twice.\n"
    )
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "" and lines:
            break
        lines.append(line)

    email_text = "\n".join(lines).strip()
    if not email_text:
        print("No email text provided. Exiting.")
        return

    response = chain.invoke({"email": email_text})
    # ChatGoogleGenerativeAI returns a BaseMessage; .content holds the text
    print("\n--- Drafted Reply ---\n")
    print(response.content)


if __name__ == "__main__":
    main()
