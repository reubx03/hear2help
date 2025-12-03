from pipeline import AssistantPipeline

def main():
    print("\n=== Railway Assistant CLI ===")

    assistant = AssistantPipeline(debug=True)

    audio = input("\nEnter path to audio file (or type text): ")
    reply = assistant.run(audio)

    print("\n====== FINAL OUTPUT ======")
    print(reply)

if __name__ == "__main__":
    main()
