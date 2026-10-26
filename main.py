import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from guardrail import validate_and_resolve
from agents.orchestrator import orchestrator


def analyze_ticker(ticker: str, log_dir: Path | None = None) -> None:
    """Run the full fundamental analysis pipeline for a verified ticker."""
    prompt = (
        f"Perform a complete fundamental analysis for the stock ticker: {ticker}\n"
        f"Follow the full workflow: gather financial data, news sentiment, and risk "
        f"assessment, then run valuation, and finally produce the investment brief."
    )

    result = orchestrator(prompt)

    if log_dir is not None:
        report = str(result)
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = log_dir / f"{ticker}_{timestamp}.txt"
        out_file.write_text(report, encoding="utf-8")
        print(f"\n[Report saved to {out_file}]")


def run(user_input: str, log_dir: Path | None = None) -> None:
    """Validate user input, resolve to a ticker, then run the pipeline."""
    print("\nValidating your request ...")
    check = validate_and_resolve(user_input)

    if not check["ok"]:
        print(f"\n{check['reply']}")
        return

    ticker = check["ticker"]
    name   = check["name"]
    print(f"Resolved: {name} ({ticker})")
    print("Starting analysis — this may take 30–60 seconds.\n")

    analyze_ticker(ticker, log_dir=log_dir)


if __name__ == "__main__":
    log_dir = None

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        run(user_input, log_dir=log_dir)
    else:
        user_input = input("What would you like to analyze? ").strip()
        while True:
            check = validate_and_resolve(user_input)
            if check["ok"]:
                print(f"\nResolved: {check['name']} ({check['ticker']})")
                print("Starting analysis — this may take 30–60 seconds.\n")
                analyze_ticker(check["ticker"], log_dir=log_dir)
                break
            else:
                print(f"\n{check['reply']}\n")
                user_input = input("> ").strip()
