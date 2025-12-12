import time
from transformers import pipeline

print("Loading models.")

#DistilBERT
model_a = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

#RoBERTa
model_b = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

#test sentence
test_text = "I absolutely love coding."
print(f"\nTest Sentence: '{test_text}'\n")

print("\nTesting.")

#test BERT
start_time = time.time()
result_a = model_a(test_text)
end_time = time.time()
time_a = end_time - start_time

print(f"DistilBERT Result: {result_a}")
print(f"Time: {time_a:.4f} seconds\n")

#test Roberta
start_time = time.time()
result_b = model_b(test_text)
end_time = time.time()
time_b = end_time - start_time

print(f"RoBERTa Result:    {result_b}")
print(f"Time: {time_b:.4f} seconds\n")

#result
print("\nRESULT\n")
if time_a < time_b:
    print(f"Speed Analysis: DistilBERT is FASTER.")
else:
    print("Speed Analysis: RoBERTa is FASTER.")
    
