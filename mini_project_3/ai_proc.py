from transformers import pipeline
from PIL import Image
import requests
import warnings 

warnings.filterwarnings("ignore")

class AIProcessor:
    def __init__(self):
        print("AI Processor is starting.")

    def sentiment_analysis(self, text):
        
        classifier = pipeline("sentiment-analysis")
        result = classifier(text)
        return result

    def question_answering(self, context, question):

        qa_model = pipeline("question-answering")
        result = qa_model(question=question, context=context)
        return result

    def zero_shot_classification(self, text, labels):
      
        classifier = pipeline("zero-shot-classification")
        result = classifier(text, candidate_labels=labels)
        return result
    
    def text_summarization(self, text):
        summarizer = pipeline("summarization")
        result = summarizer(text, max_length=35, min_length=10, do_sample=False)
        return result
    
    def text_generation(self, text):
        generator = pipeline("text-generation")
        result = generator(text, max_new_tokens=10, truncation=True, pad_token_id=50256)
        return result
    
    def text_translation(self, text):
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
        result = translator(text)
        return result
    
    def mask_filling(self, text):
        filler = pipeline("fill-mask")
        result = filler(text)
        return result
    
    def image_classification(self, image):
        classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
        result = classifier(image)
        return result
    
    def named_entity_recognition(self, text):
        ner = pipeline("ner", grouped_entities=True)
        result = ner(text)
        return result
    
    def automatic_speech_recognition(self, audio):
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
        result = asr(audio)
        return result

#run
if __name__ == "__main__":
    ai = AIProcessor() #start class

    print("\nSentiment Analysis")
    res = ai.sentiment_analysis("I love coding.")
    print(f"Input: I love coding.")
    print(f"Output: {res[0]['label']} (Score: {res[0]['score']:.2f})")

    print("\nQuestion Answering")
    context_text = "I use Python language and VSC application."
    question_text = "Which coding language do you use?"
    res = ai.question_answering(context_text, question_text)
    print(f"Question: {question_text}")
    print(f"Answer: {res['answer']}")

    print("\nZero-shot Classification")
    text_zs = "I need a new laptop."
    labels_zs = ["politics", "technology", "sports"]
    res = ai.zero_shot_classification(text_zs, labels_zs)
    print(f"Input: {text_zs}")
    print(f"Prediction: {res['labels'][0]} (Confidence: {res['scores'][0]:.2f})")

    print("\nText Summarization")
    long_text = """Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
    that are programmed to think like humans and mimic their actions. The term may also be applied to any 
    machine that exhibits traits associated with a human mind such as learning and problem-solving. 
    The ideal characteristic of artificial intelligence is its ability to rationalize and take actions 
    that have the best chance of achieving a specific goal. A subset of artificial intelligence is 
    machine learning, which refers to the concept that computer programs can automatically learn from 
    and adapt to new data without being assisted by humans."""
    res = ai.text_summarization(long_text)
    print(f"Summary: {res[0]['summary_text']}")

    print("\nText Generation")
    start_text = "Software development will change the world by"
    res = ai.text_generation(start_text)
    print(f"Generated: {res[0]['generated_text']}")

    print("\nText Translate")
    res = ai.text_translation("Hello world!")
    print(f"Translation: {res[0]['translation_text']}")

    print("\nMask Filling")
    mask_text = "<mask> is the capital of Turkey."
    res = ai.mask_filling(mask_text)
    print(f"Input: {mask_text}")
    print(f"Top Prediction: {res[0]['token_str']} (Score: {res[0]['score']:.2f})")

    print("\nImage Classification")
    try:
        url = "https://plus.unsplash.com/premium_photo-1694819488591-a43907d1c5cc?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8Y3V0ZSUyMGRvZ3xlbnwwfHwwfHx8MA%3D%3D"
        image = Image.open(requests.get(url, stream=True).raw)
        res = ai.image_classification(image)
        print(f"Prediction: {res[0]['label']} ({res[0]['score']:.2f})")
    except Exception as e:
        print(f"Image Error: {e}")

    print("\nNamed Entity Recognition")
    ner_text = "Elon Musk works at Tesla in California."
    res = ai.named_entity_recognition(ner_text)
    print(f"Input: {ner_text}")
    for entity in res:
        print(f" - {entity['word']}: {entity['entity_group']}")
        
    print("\nAutomatic Speech Recognition")
    try:
        res = ai.automatic_speech_recognition("speech_Rec.wav") 
        print(f"Transcription: {res['text']}")
    except Exception as e:
        print(f"Error: {e}")

