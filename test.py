import pyttsx3

def text_to_speech(text):
    engine = pyttsx3.init()

    # Set speech rate and volume
    engine.setProperty('rate', 175)  
    engine.setProperty('volume', 1)  

    # Use system default voice without explicitly selecting any
    try:
        engine.say(text)
        # Wait for the speech synthesis to finish
        engine.runAndWait()
    except Exception as e:
        print(f"Error during speech synthesis: {e}")

# Example usage
output_text = """
The person who crashed into two scooters and fled the scene will face several charges, as outlined in the provided legal notice:

* Hit and Run under Section 161 of the Motor Vehicles Act, 1988: This is a serious offense for leaving the scene of an accident without providing assistance or reporting it to the authorities.
* Failure to report the accident as required under Section 134: This charge is related to the driver's obligation to report the accident to the nearest police station.
* Negligent driving causing injury under Section 279 of the Indian Penal Code (IPC): This charge applies because the driver's actions caused injury to the scooter riders.

These charges can result in significant penalties, including fines, imprisonment, or both. Additionally, the victims may initiate a separate inquiry into potential civil liabilities.
"""

text_to_speech(output_text)
