import json

# Example usage
from src.sentiment_intention_analyzer import SentimentIntentionAnalyzer


def load_transcript_from_file(file_path):
    with open(file_path, "r") as file:
        transcript = json.load(file)
    return transcript


def save_transcript_to_file(transcript, file_path):
    with open(file_path, "w") as file:
        file.write(transcript)
        #json.dump(transcript, file, indent=2)


def format_output(step, sentiment, sent_score, intent, intent_score):
    formatted_output = f"Step {step} - Sentiments: {sentiment} with score: {sent_score}\nIntent: {intent} with score: {intent_score}"
    return formatted_output


if __name__ == "__main__":
    analyzer = SentimentIntentionAnalyzer()
    transcript_path = "data/transcript.json"
    transcript = load_transcript_from_file(transcript_path)
    with open(transcript_path, "r") as file:
        transcript = json.load(file)

    result = ""
    for dialog in transcript:
        predicted_sentiment,predicted_sentiment_score,predicted_intent,predicted_intent_score = analyzer.analyze_transcript(dialog['buyer'] + "\n" + dialog['seller'])
        result += format_output(
            dialog["step"],
            predicted_sentiment,
            predicted_sentiment_score,
            predicted_intent,
            predicted_intent_score,
        )
        result += '\n'
   
    print(result)
    save_transcript_to_file(result, 'data/classified_dialog.txt')

