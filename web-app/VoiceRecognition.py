import speech_recognition as sr

def recognize_voice_command():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Listening... Please speak into the microphone.")
        
        # Debugging: Uncomment if you wish to test adjustment for noise
        # recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            # Increase timeout or remove this timeout for testing
            audio = recognizer.listen(source, timeout=10)  
            print("Audio captured, processing...")
            
            # Recognize the speech using Google Web Speech API
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            return command
        
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start.")
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said. Please try again.")
        except sr.RequestError as e:
            print(f"Could not request results from the speech recognition service; {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    while True:
        command = recognize_voice_command()
        if command:
            # Exit the loop if the user says "exit" or "quit"
            if command.lower() in ["exit", "quit"]:
                print("Exiting the program. Goodbye!")
                break
            # Handle other commands here
            print(f"Processing command: {command}")
