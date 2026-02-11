from pipeline import AssistantPipeline

def main():
    print("\n=== Railway Assistant CLI ===")

    assistant = AssistantPipeline(debug=True)

    test=''
    audio = input("\nEnter path to audio file (or type text): ")
    if(audio=='hin' or audio=='mal'):
        print('\n using test audio...')
        audio=f"C:\\Coding Projects\\hear2help WiP\\audio\\{audio}.mp3"
    reply = assistant.run(audio)

    print("\n====== FINAL OUTPUT ======")
    print(reply)

if __name__ == "__main__":
    main()
