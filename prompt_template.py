import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

client =anthropic.Anthropic()

class PromptTemplate:
    def __init__(self, system: str):
        self.system = system
        self.model = "claude-haiku-4-5-20251001"

    def count_tokens(self,user_message) -> int:
       response = client.messages.count_tokens(
           model =self.model,
           system = self.system,
           messages = [
               {"role": "user", "content": user_message}
           ]
       )
       return response.input_tokens
    
    def run(self, user_message: str, max_tokens: int = 256) -> dict:
        message  = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        block = message.content[0]
        if isinstance(block, anthropic.types.TextBlock):
            response_text = block.text.replace("\n", " ").strip()
        else:
            response_text = str(block)

        return {
            "response": response_text,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        }


templates = [
    {
        "system": "You are a senior .NET architect. Answer questions about software design concisely.",
        "user": "When should I use a microservice instead of a monolith?",
    },
    {
        "system": "You are a JSON API. Respond ONLY with valid JSON. No explanation.",
        "user": "Give me 10 athletes with fields: name, sports.",
    },
    {
        "system": "You are a tough technical interviewer. Ask ONE follow-up question to probe deeper.",
        "user": "I built a RAG system using AWS Bedrock Knowledge Bases.",
    },
]

for i,t in enumerate(templates,1):
    tmpl  = PromptTemplate(system=t["system"])
    result = tmpl.run(user_message=t["user"])

    print(f"─── Template {i} ──────────────────────────────────────────")
    print(f"System  : {t['system'][:60]}...")
    print(f"User    : {t['user']}")
    print(f"Response: {result['response']}")
    print(f"Tokens  : {result['input_tokens']} in / {result['output_tokens']} out / {result['total_tokens']} total")
    print()