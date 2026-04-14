import gradio as gr
from app.rag_pipeline import ask_question


def interface(file, query):
    file_path = file if isinstance(file, str) else file.name
    return ask_question(file_path, query)


demo = gr.Interface(
    fn=interface,
    inputs=[
        gr.File(label="Upload PDF", type="filepath"),
        gr.Textbox(label="Ask a question"),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="QA Bot using RAG",
    description="Upload a PDF and ask questions about its content.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
