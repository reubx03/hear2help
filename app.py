from pipeline import AssistantPipeline

def main():
    print("\n=== Railway Assistant CLI ===")

    assistant = AssistantPipeline(debug=True)

    test=''
    audio = input("\nEnter path to audio file (or type text): ")
    if(audio=='test'):
        print('\n using test audio...')
        audio="C:\\Coding Projects\\hear2help WiP\\audio\\testhin.wav"
    reply = assistant.run(audio)

    print("\n====== FINAL OUTPUT ======")
    print(reply)

if __name__ == "__main__":
    main()
