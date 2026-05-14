import anthropic, json
from dotenv import load_dotenv
load_dotenv()
client = anthropic.Anthropic()

SENTIMENT_TOOL = {
    "name": "record_sentiment",
    "description": "Record the sentiment analysis result for a product review.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "mixed"],
                "description": "Overall sentiment"
            },
            "score": {
                "type": "number",
                "description": "Confidence score from 0.0 to 1.0"
            },
            "key_issue": {
                "type": "string",
                "description": "The main positive or negative point, or null if none"
            }
        },
        "required": ["sentiment", "score", "key_issue"]
    }
}

def extract_via_tool(review: str) -> dict:
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        tools=[SENTIMENT_TOOL],
        tool_choice={"type": "tool", "name": "record_sentiment"},
        messages=[{"role": "user", "content": f"Analyze this review: {review}"}]
    )
    # The model MUST return a tool_use block
    tool_block = next(b for b in msg.content if b.type == "tool_use")
    return tool_block.input  # already a dict, no json.loads() needed

def extract_via_prompt(review: str) -> dict:
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system="""Extract sentiment data from product reviews.
Respond ONLY with valid JSON in this exact format:
{"sentiment": "positive|negative|mixed", "score": 0.0-1.0, "key_issue": "string or null"}""",
        messages=[{"role": "user", "content": review}]
    )
    block = msg.content[0]
    if not isinstance(block, anthropic.types.TextBlock):
        raise ValueError(f"Unexpected block type: {type(block)}")
    text = block.text.strip()
    print("Raw response:", repr(text))  # debug — remove once working
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    return json.loads(text)


TEST_REVIEWS = [
    "Works great, very fast delivery.",
    "Stopped working after one week. Terrible.",
    "Good features but the battery is awful.",
    "Screen is beautiful but the software crashes constantly.",
    "Exactly as described. Happy with it.",
]

if __name__ == "__main__":
    print(f"{'Review':<45} {'Method':<10} {'Sentiment':<10} {'Score':>6} {'Key Issue'}")
    print("-" * 100)

    for review in TEST_REVIEWS:
        short = review[:42]

        try:
            r1 = extract_via_prompt(review)
            print(f"{short:<45} {'prompt':<10} {r1['sentiment']:<10} {r1['score']:>6.2f}  {r1.get('key_issue','')}")
        except Exception as e:
            print(f"{short:<45} {'prompt':<10} ERROR: {e}")

        r2 = extract_via_tool(review)
        print(f"{short:<45} {'tool':<10} {r2['sentiment']:<10} {r2['score']:>6.2f}  {r2.get('key_issue','')}")
        print()