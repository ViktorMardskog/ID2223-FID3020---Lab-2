
import gradio as gr
from llama_cpp import Llama

from huggingface_hub import hf_hub_download
import os
#https://llama-cpp-python.readthedocs.io/en/latest/api-reference/

model_path = hf_hub_download(
    repo_id="ViktorMardskog/lora_model_3e-4LR_10k_1b_val",
    filename="llama-3.2-1b-instruct.Q4_K_M.gguf",
    repo_type="model",
) 

llm = Llama(model_path=model_path, n_ctx=500,  n_gpu_layers=0, verbose=False)

#to set brainstoorming model to same, uncomment this and comment the other.
""" model_path_B = hf_hub_download(
    repo_id="ViktorMardskog/lora_model_3e-4LR_10k_1b_val",
    filename="llama-3.2-1b-instruct.Q4_K_M.gguf",
    repo_type="model",
) """


model_path_B = hf_hub_download(
    repo_id="ViktorMardskog/lora_model_3e-4LR_1b_val_BrainStoorming",
    filename="llama-3.2-1b-instruct.Q4_K_M.gguf",
    repo_type="model",
) 

llm_brainstoorming = Llama(model_path=model_path_B, n_ctx=500,  n_gpu_layers=0, verbose=False)




def evaluate(expanded_idea):
    messages = [{"role": "system", "content": "You are an assistant."}]

    message =f"Idea to be evaluated: {expanded_idea}.\n Evaluate the idea with a score (1–10) and give 3 concrete concerns.\n Format:\n Score: X/10\n Concerns:\n 1. ...\n 2. ...\n 3. ..."

    messages.append({"role": "user", "content": message})

    #This code snippet below is directly taken from the UI chatbot template in huggingface (gradio)
    response = ""

    for message_part in llm.create_chat_completion(
        messages,
        max_tokens=128,
        stream=True,
        temperature=0.8,
        top_p=0.9,
    ):
        choices = message_part['choices']
        token = ""
        if choices and choices[0]["delta"].get("content"):
            token =choices[0]["delta"]["content"]
        response += token
        yield response

def refine(expanded_idea):
    messages = [{"role": "system", "content": "You are an assistant."}]

    message =f"Refine the following idea: {expanded_idea}."

    messages.append({"role": "user", "content": message})

    #This code snippet below is directly taken from the UI chatbot template in huggingface (gradio)
    response = ""

    for message_part in llm.create_chat_completion(
        messages,
        max_tokens=128,
        stream=True,
        temperature=0.8,
        top_p=0.9,
    ):
        choices = message_part['choices']
        token = ""
        if choices and choices[0]["delta"].get("content"):
            token =choices[0]["delta"]["content"]
        response += token
        yield response

def brainstorm(expanded_idea):
    messages = [{"role": "system", "content": "You are an assistant."}]

    message =f"Idea Input: {expanded_idea}. Brainstorm three features this idea could include. Reply ONLY with bullet-points.\n"

    messages.append({"role": "user", "content": message})

    #This code snippet below is directly taken from the UI chatbot template in huggingface (gradio)
    response = ""

    for message_part in llm_brainstoorming.create_chat_completion(
        messages = messages,
        max_tokens=128,
        stream=True,
        temperature=0.8,
        top_p=0.9,
    ):
        choices = message_part['choices']
        token = ""
        if choices and choices[0]["delta"].get("content"):
            token =choices[0]["delta"]["content"]
        response += token
        yield response


def add_note(notes,title):
    if notes is None:
        notes = []

    title =title.strip()

    notes = notes +[{"title": title, "text": ""}]
    choices = [n["title"] for n in notes]

    return gr.update(choices=choices, value=title), notes,""

def select_note(notes, selected_title):
    if notes is None:
        notes = []
    if not selected_title:
        return "", notes

    for n in notes:
        if n["title"] == selected_title:
            return n["text"], notes

    return "", notes

def update_note_text(notes_state, selected_title, new_expanded_idea):
    if not selected_title:
        return notes_state
    if notes_state is None:
        notes_state = []
    
    new_notes= []
    for note in notes_state:
        if note["title"] == selected_title:
            new_notes.append({"title": note["title"], "text": new_expanded_idea})
        else:
            new_notes.append(note)
    return new_notes

def delete_note(notes, selected_title):
    if notes is None: notes = []
    if not selected_title:
        #then keep everything like before because not selected any for deletion
        choices = [n["title"] for n in notes]
        return gr.update(choices=choices,value=None), notes

    new_notes =[]
    for note in notes:
        if note["title"]!= selected_title:
            new_notes.append(note)
    
    choices = []
    for new_note in new_notes:
        choices.append(new_note["title"])

    return gr.update(choices=choices, value=None),new_notes



def append_LLM_text(expanded_idea, llm_output):
    if not llm_output: return expanded_idea
    if not expanded_idea: return llm_output

    return expanded_idea + "\n" + llm_output

def replace_idea_text(expanded_idea, llm_output):
    new =llm_output or expanded_idea
    return new

with gr.Blocks(title="Brainstorming helper") as demo:
    notes_state = gr.State([])  #title --> body

    with gr.Row():

        with gr.Column(scale=2):
            gr.Markdown("Sticky notes")
            notes_list = gr.Radio(choices=[], label="Ideas", interactive=True)
            new_title = gr.Textbox(label="New idea title", placeholder="Short title for the idea")
            add_note_btn = gr.Button("Add sticky note")
            delete_btn = gr.Button("Delete selected note")

        with gr.Column(scale=3):
            expanded_idea = gr.Textbox(label="Expanded idea", lines=12, placeholder="Describe the idea..(stickynote has to be selected)")
            with gr.Row():
                save_btn = gr.Button("Save idea")
            with gr.Row():
                refine_btn = gr.Button("Refine idea")
                brainstorm_btn = gr.Button("Brainstorm around idea")
                evaluate_btn = gr.Button("Evaluate idea")

            llm_output_box=gr.Textbox(label="LLM output",lines=10,interactive=False)
            with gr.Row():
                append_btn = gr.Button("Append to expanded idea")
                replace_btn = gr.Button("Replace expanded idea")

    save_btn.click(update_note_text, inputs=[notes_state, notes_list, expanded_idea], outputs=[notes_state])

    refine_btn.click(refine, inputs=expanded_idea, outputs=llm_output_box)
    brainstorm_btn.click(brainstorm, inputs=expanded_idea, outputs=llm_output_box)
    evaluate_btn.click(evaluate, inputs=expanded_idea, outputs=llm_output_box)

    
    add_note_btn.click(add_note, inputs=[notes_state, new_title],outputs=[notes_list, notes_state, expanded_idea])
    notes_list.change(select_note, inputs=[notes_state, notes_list],outputs=[expanded_idea, notes_state])
    delete_btn.click(delete_note, inputs=[notes_state, notes_list], outputs=[notes_list, notes_state])

    append_btn.click(append_LLM_text,inputs=[expanded_idea, llm_output_box], outputs=[expanded_idea])

    replace_btn.click(replace_idea_text, inputs=[expanded_idea, llm_output_box], outputs=[expanded_idea])


if __name__ == "__main__":
    demo.queue()
    demo.launch()













