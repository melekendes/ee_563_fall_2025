import gradio as gr
from ai_proc import AIProcessor 

ai = AIProcessor()

def func_sentiment(text):
    result = ai.sentiment_analysis(text)
    label = result[0]['label']
    score = result[0]['score']
    return f"Label: {label}\nScore: %{score*100:.2f}"

def func_qa(context, question):
    result = ai.question_answering(context, question)
    return f"Answer: {result['answer']}\n(Score: {result['score']:.2f})"

def func_zeroshot(text, labels):
    candidate_labels = [label.strip() for label in labels.split(",")]
    result = ai.zero_shot_classification(text, candidate_labels)
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    return f"Class: {top_label} (%{top_score*100:.2f})"

def func_summarize(text):
    result = ai.text_summarization(text)
    return result[0]['summary_text']

def func_generate(text):
    result = ai.text_generation(text)
    return result[0]['generated_text']

def func_translate(text):
    result = ai.text_translation(text)
    return result[0]['translation_text']

def func_mask(text):
    result = ai.mask_filling(text)
    output = ""
    for item in result[:2]: 
        output += f"Prediction: {item['token_str']} (Score: {item['score']:.2f})\n"
    return output

def func_image(image):
    if image is None:
        return "Upload an image."
    result = ai.image_classification(image)
    top_result = result[0]
    return f"{top_result['label']} (%{top_result['score']*100:.2f})"

def func_ner(text):
    result = ai.named_entity_recognition(text)
    output = ""
    for entity in result:
        group = entity.get('entity_group', entity.get('entity')) 
        word = entity['word']
        output += f"{word} -> {group}\n"
    return output

def func_asr(audio_path):
    if audio_path is None:
        return "No audio file or recording found."
    result = ai.automatic_speech_recognition(audio_path)
    return result['text']

#Interface Design
with gr.Blocks(title="Large Language Models (LLMs) with HuggingFace Experiments") as demo:
    gr.Markdown("Large Language Models (LLMs) with HuggingFace Experiments")
    gr.Markdown("# EE563 - AI in Practice (Mini Project #3)")


    with gr.Tabs():
        
        with gr.TabItem("Sentiment Analysis"):
            with gr.Row():
                with gr.Column():
                    sent_input = gr.Textbox(label="Input Text", lines=6)
                    sent_btn = gr.Button("Analyze")
                with gr.Column():
                    sent_output = gr.Textbox(label="Result", lines=6)
            sent_btn.click(func_sentiment, inputs=sent_input, outputs=sent_output)

        with gr.TabItem("Question Answering"):
            with gr.Row():
                with gr.Column():
                    qa_context = gr.Textbox(label="Context", lines=6)
                    qa_question = gr.Textbox(label="Question")
                    qa_btn = gr.Button("Answer")
                with gr.Column():
                    qa_output = gr.Textbox(label="Result", lines=6)
            qa_btn.click(func_qa, inputs=[qa_context, qa_question], outputs=qa_output)

        with gr.TabItem("Zero-Shot Classification"):
            zs_input = gr.Textbox(label="Input Text", lines=6)
            zs_labels = gr.Textbox(label="Candidate Labels (comma separated)", lines=6)
            zs_btn = gr.Button("Classify")
            zs_output = gr.Textbox(label="Result")
            zs_btn.click(func_zeroshot, inputs=[zs_input, zs_labels], outputs=zs_output)

        with gr.TabItem("Summarization"):
            summ_input = gr.Textbox(label="Long Text", lines=10)
            summ_btn = gr.Button("Summarize")
            summ_output = gr.Textbox(label="Summary", lines=6)
            summ_btn.click(func_summarize, inputs=summ_input, outputs=summ_output)

        with gr.TabItem("Text Generation"):
            gen_input = gr.Textbox(label="Starting Sentence", lines=6)
            gen_btn = gr.Button("Generate")
            gen_output = gr.Textbox(label="Generated Text", lines=6)
            gen_btn.click(func_generate, inputs=gen_input, outputs=gen_output)

        with gr.TabItem("Translation (En->Fr)"):
            trans_input = gr.Textbox(label="English Text", lines=6)
            trans_btn = gr.Button("Translate")
            trans_output = gr.Textbox(label="French Translation", lines=6)
            trans_btn.click(func_translate, inputs=trans_input, outputs=trans_output)

        with gr.TabItem("Mask Filling"):
            mask_input = gr.Textbox(label="Masked Sentence (use <mask>)", lines=6)
            mask_btn = gr.Button("Predict")
            mask_output = gr.Textbox(label="Predictions", lines=6)
            mask_btn.click(func_mask, inputs=mask_input, outputs=mask_output)

        with gr.TabItem("Image Classification"):
            img_input = gr.Image(type="pil", label="Upload Image")
            img_btn = gr.Button("Classify")
            img_output = gr.Textbox(label="Result", lines=6)
            img_btn.click(func_image, inputs=img_input, outputs=img_output)

        with gr.TabItem("Named Entity Recognition"):
            ner_input = gr.Textbox(label="Input Text", lines=6)
            ner_btn = gr.Button("Analyze")
            ner_output = gr.Textbox(label="Entities", lines=6)
            ner_btn.click(func_ner, inputs=ner_input, outputs=ner_output)

        with gr.TabItem("Automatic Speech Recognition"):
            asr_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or Upload Audio")
            asr_btn = gr.Button("Transcribe")
            asr_output = gr.Textbox(label="Transcription", lines=6)
            asr_btn.click(func_asr, inputs=asr_input, outputs=asr_output)

if __name__ == "__main__":
    demo.launch()